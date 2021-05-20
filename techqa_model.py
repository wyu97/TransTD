import argparse
import glob
import logging
import os
from os import path
import random
import timeit
import json
from functools import partial
import numpy as np
import torch
import gc
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,

    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,

    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,

    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)

from techqa_metrics import predict_output
from techqa_processor import load_and_cache_examples

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class BertLargeClassifier(nn.Module):

    def __init__(self, hidden):
        super(BertLargeClassifier, self).__init__()

        self.output_layer = nn.Sequential(nn.Linear(hidden, 256),
                                          torch.nn.Dropout(0.5),
                                          nn.ReLU(True),
                                          nn.Linear(256, 2))

    def forward(self, x):
        return self.output_layer(x)


class AlBertLargeClassifier(nn.Module):

    def __init__(self, hidden):
        super(AlBertLargeClassifier, self).__init__()

        self.output_layer = nn.Sequential(nn.Linear(hidden, 256),
                                          torch.nn.Dropout(0.5),
                                          nn.ReLU(True),
                                          nn.Linear(256, 2))

    def forward(self, x):
        return self.output_layer(x)


class MulBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

        self.dr_outputs = BertLargeClassifier(config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, snippet_length=None, document_pair=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence output is shaped as [batch, 512, 1024]
        sequence_output = outputs[0]

        ''' calculate RC outputs '''
        # logits is shaped as [batch, 512, 2]
        rc_logits = self.qa_outputs(sequence_output)

        # start_logits and end_logits are shaped as [batch, 512, 1] (both)
        start_logits, end_logits = rc_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        ''' calculate DR outputs '''
        cls_tokens = torch.mean(outputs[0], dim=1)
        # cls_tokens = outputs[0][:, 0, :]

        assert torch.equal(outputs[0], outputs[2][-1]) == True

        dr_logits = self.dr_outputs(cls_tokens)

        output = [start_logits, end_logits, dr_logits]

        if start_positions is not None and end_positions is not None:

            ''' calculate rc loss '''
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            rc_loss = (start_loss + end_loss) / 2

            ''' calculate dr loss '''
            loss_fct_dr = CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).to('cuda'))

            dr_loss = loss_fct_dr(dr_logits, document_pair)

            ''' calculate rc weight term '''

            soft_start, soft_end = F.softmax(start_logits, dim=1), F.softmax(end_logits, dim=1)

            span_truth = (end_positions - start_positions).float()
            span_truth[span_truth <= 0] = float("inf")

            # option 1: softmax and take argmax one
            (_, max_start), (_, max_end) = torch.max(soft_start, dim=1), torch.max(soft_end, dim=1)
            span_pred = torch.abs((max_end - max_start) - (end_positions - start_positions)).float()

            weight = torch.exp(torch.min((span_pred/span_truth).float(), torch.ones(span_pred.shape).to('cuda')))

            ''' calculate final loss '''
            total_loss = weight * rc_loss + 4 * dr_loss

            output = [total_loss,] + output

        return output  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class MulAlbertForQuestionAnswering(AlbertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

        self.dr_outputs = AlBertLargeClassifier(config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, snippet_length=None, document_pair=None):

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence output is shaped as [batch, 512, 1024]
        sequence_output = outputs[0]

        ''' calculate RC outputs '''
        # logits is shaped as [batch, 512, 2]
        rc_logits = self.qa_outputs(sequence_output)

        # start_logits and end_logits are shaped as [batch, 512, 1] (both)
        start_logits, end_logits = rc_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        ''' calculate DR outputs '''
        cls_tokens = torch.mean(outputs[0], dim=1)
        # cls_tokens = outputs[0][:, 0, :]

        assert torch.equal(outputs[0], outputs[2][-1]) == True

        dr_logits = self.dr_outputs(cls_tokens)

        output = [start_logits, end_logits, dr_logits]

        if start_positions is not None and end_positions is not None:

            ''' calculate rc loss '''
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            rc_loss = (start_loss + end_loss) / 2

            ''' calculate dr loss '''
            loss_fct_dr = CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).to('cuda'))

            dr_loss = loss_fct_dr(dr_logits, document_pair)

            ''' calculate rc weight term '''

            soft_start, soft_end = F.softmax(start_logits, dim=1), F.softmax(end_logits, dim=1)

            span_truth = (end_positions - start_positions).float()
            span_truth[span_truth <= 0] = float("inf")

            # option 1: softmax and take argmax one
            (_, max_start), (_, max_end) = torch.max(soft_start, dim=1), torch.max(soft_end, dim=1)
            span_pred = torch.abs((max_end - max_start) - (end_positions - start_positions)).float()

            # option 2: calculate start and end expection
            # TODO TODO TODO

            weight = torch.exp(torch.min((span_pred/span_truth).float(), torch.ones(span_pred.shape).to('cuda')))

            # print(weight)
            ''' calculate final loss '''
            # print(rc_loss, dr_loss)
            total_loss = rc_loss + 3 * dr_loss
            # total_loss = weight * rc_loss + 3 * dr_loss

            output = [total_loss,] + output

        return output  # (loss), start_logits, end_logits, (hidden_states), (attentions)
