import heapq
import json
import logging
import os
from collections import defaultdict

import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from typing import List, Dict, Optional, Callable, Iterable

from techqa_evaluation import evaluate
from techqa_processor import TechQaInputFeature

from torch.utils.data import TensorDataset

_NEGATIVE_INFINITY = float('-inf')

@functools.total_ordering
class TechQAPrediction:
    def __init__(self, doc_id, start_offset, end_offset, score, is_doc, start_score, end_score):
        self.doc_id = doc_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.score = score
        self.is_doc = is_doc
        self.start_score = start_score
        self.end_score = end_score

    def __hash__(self):
        return hash((self.doc_id, self.start_offset, self.end_offset, self.score, self.is_doc, self.start_score, self.end_score))

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score


class BestSpanTracker(object):
    __slots__ = ['n_logits_to_search', 'max_spans_to_track', 'top_n_spans', 'max_answer_length', 'score_calculator']

    def __init__(self, n_logits_to_search: int, max_spans_to_track: int, max_answer_length: int,
                 score_calculator: Callable):

        self.n_logits_to_search = n_logits_to_search
        self.max_spans_to_track = max_spans_to_track
        self.max_answer_length = max_answer_length
        self.top_n_spans = list()
        self.score_calculator = score_calculator

    def collect_best_non_null_spans(self, start_logits: List[float], end_logits: List[float],
                                    dr_logits: List[float], feature: TechQaInputFeature):

        null_span_score = end_logits[0] + start_logits[0]

        for start_idx, end_idx in self._get_start_end_scores_for_possible_spans(
                get_best_indexes(start_logits, self.n_logits_to_search),
                get_best_indexes(end_logits, self.n_logits_to_search), feature):

            self._update_top_n_spans(
                TechQAPrediction(
                    doc_id=feature.doc_id,
                    start_offset=feature.token_to_original_start_offset[start_idx],
                    end_offset=feature.token_to_original_end_offset[end_idx],
                    start_score=start_logits[start_idx],
                    end_score=end_logits[end_idx],
                    score=self.score_calculator(span_score=end_logits[end_idx] + start_logits[start_idx], null_span_score=null_span_score, dr_logits=dr_logits),
                    is_doc=dr_logits[1]))

    def _update_top_n_spans(self, prediction: TechQAPrediction):
        if len(self.top_n_spans) < self.max_spans_to_track:
            # Collect up to nbest_size spans
            heapq.heappush(self.top_n_spans, prediction)
        elif prediction.score > self.top_n_spans[0].score:
            # This span has a better score than the ones we've seen, so swap it into the set
            heapq.heapreplace(self.top_n_spans, prediction)

    def get_nbest_spans(self, n: int) -> List[TechQAPrediction]:
        return heapq.nlargest(n=n, iterable=self.top_n_spans)

    def _get_start_end_scores_for_possible_spans(self, start_indeces, end_indeces, feature):
        for start_index in start_indeces:
            for end_index in end_indeces:
                # This should be an actual span from the passage tokens, make sure that's
                # actually the case. We could hypothetically create invalid predictions,
                # e.g., predict that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_original_start_offset:
                    continue
                if end_index not in feature.token_to_original_end_offset:
                    continue
                if end_index < start_index:
                    continue
                # We may have shuffled sentences during feature prep, so account for that
                if feature.token_to_original_start_offset[start_index] > \
                        feature.token_to_original_end_offset[
                            end_index]:
                    continue
                length = end_index - start_index + 1
                if length > self.max_answer_length:
                    continue
                yield start_index, end_index

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def _save_predictions_to_output_dir(threshold, predictions, output_dir, epoch=-1):

    if epoch < 0:
        output_prediction_file = os.path.join(output_dir, "predictions.json")
    else:
        output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(epoch))

    with open(output_prediction_file, "w") as outF:
        logging.info('Saving %d predictions to %s' % (len(predictions), output_prediction_file))
        json.dump({'threshold': threshold, 'predictions': predictions}, outF, indent=2)

def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1],
                             reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def compute_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float, dr_logits: List[float], *args, **kwargs):

    # return span_score
    return span_score - null_span_score
    # return dr_logits[1] * span_score - null_span_score


def calculate_precision_recall_of_document_pairness(y_pred: List[float], y_true: List[float]):

    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix

    macro_precision = round(precision_score(y_true, y_pred, average='macro'), 4)
    macro_recall = round(recall_score(y_true, y_pred, average='macro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, average='micro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 4)
    confusion_matrix = confusion_matrix(y_true, y_pred).tolist()

    return micro_f1, macro_precision, macro_recall, macro_f1, confusion_matrix


def calculate_document_pariness_by_cls_prediction(
        cls_prediction: TensorDataset, document_pairness: TensorDataset):

    max_value, max_index = torch.max(cls_prediction, dim=1)
    return max_index.tolist(), document_pairness.tolist()


def predict_output(device, eval_features: List[TechQaInputFeature], eval_dataset: TensorDataset,
                   gold_dict: Dict, corpus: Dict, model: nn.Module, model_type: str,
                   nbest_size: int, max_answer_length: int,
                   output_dir: str, predict_batch_size: int,
                   threshold: Optional[float] = None,
                   n_to_predict: Optional[int] = 1, epoch: Optional[int] = -1):

    logging.info("**** Starting Predict ****")
    logging.info("\tNum examples = %d", len(gold_dict))
    logging.info("\tn_best_size = %d" % nbest_size)
    logging.info("\tn_to_predict = %d" % n_to_predict)
    logging.info('\tpredict batch size = %d' % predict_batch_size)
    logging.info('\tdefault threshold = %s' % threshold)
    logging.info("\tmodel type = %s" % model_type)
    model.eval()

    nbest_spans_by_example_id = defaultdict(
        lambda: BestSpanTracker(max_answer_length=max_answer_length,
                                n_logits_to_search=nbest_size, max_spans_to_track=n_to_predict,
                                score_calculator=compute_score_diff_between_span_and_cls))

    progress_bar = tqdm(total=len(gold_dict), desc="Collecting best spans per question id")

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size)

    preds, trues = [], []

    for idx, batch in enumerate(eval_dataloader):

        batch = tuple(t.to(device) for t in batch)

        num_examples_so_far = len(nbest_spans_by_example_id)

        with torch.no_grad():

            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }

            outputs = model(**inputs)
            ''' softmax prob of document retrieval '''
            outputs[-1] = F.softmax(outputs[-1], dim=1)

            ''' calculate predictions and turth '''
            pred, true = calculate_document_pariness_by_cls_prediction(outputs[-1], batch[6])
            preds, trues = preds + pred, trues + true

            feature_vector_indeces = batch[7]

        # feature_vector_indeces length is 16
        for i, feature_index in enumerate(feature_vector_indeces):
            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits, dr_logits = output

            input_feature = eval_features[feature_index.item()]

            nbest_spans_by_example_id[input_feature.qid].collect_best_non_null_spans(
                start_logits=start_logits,
                end_logits=end_logits,
                dr_logits=dr_logits,
                feature=input_feature)

        progress_bar.update(n=len(nbest_spans_by_example_id) - num_examples_so_far)

    progress_bar.close()

    mi_f1, pre, rec, ma_f1, cf = calculate_precision_recall_of_document_pairness(preds, trues)

    with open(output_dir + '_ps.txt', 'a') as file_log_document_pairness:
        file_log_document_pairness.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, mi_f1, pre, rec, ma_f1, cf))

    predictions = dict()
    for query_id, nbest_spans_tracker in tqdm(nbest_spans_by_example_id.items(),
                                              'Formatting predictions into eval format'):
        predictions[query_id] = _generate_predictions(nbest_spans_tracker=nbest_spans_tracker,
                                                      n_to_predict=n_to_predict, corpus=corpus)

    logging.info('Evaluating predictions')
    if threshold is None:
        evaluation_scores = evaluate(preds=predictions, dataset=gold_dict)
        threshold = evaluation_scores['Best_QA_F1_Threshold']
    else:
        logging.info('Using pre-configured threshold: %s' % threshold)
        evaluation_scores = evaluate(preds=predictions, dataset=gold_dict, threshold=threshold)

    logging.info('Done evaluating:\n%s' % evaluation_scores)
    _save_predictions_to_output_dir(threshold=threshold,
                                    predictions=predictions, output_dir=output_dir, epoch=epoch)

    return evaluation_scores


def _generate_predictions(nbest_spans_tracker: BestSpanTracker, corpus: Dict,
                          n_to_predict: int) -> List[Dict]:

    # top_scans consists of best n predictions
    top_spans = nbest_spans_tracker.get_nbest_spans(n=n_to_predict)

    if len(top_spans) < 1:
        return [{'doc_id': '', 'answer': '', 'score': 0}]

    return [{'doc_id': prediction.doc_id,
             'answer': corpus[prediction.doc_id]['text'][
                       prediction.start_offset:prediction.end_offset],
             'start_offset': prediction.start_offset,
             'end_offset': prediction.end_offset,
             'start_score': prediction.start_score,
             'end_score': prediction.end_score,
             'score': prediction.score,
             'is_doc': prediction.is_doc} for prediction in top_spans]