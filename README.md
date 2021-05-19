# TransTD data and code

This repository contains the code package for the EMNLP'20 paper:

**[Technical Question Answering across Tasks and Domains](https://arxiv.org/pdf/2010.09780.pdf).** [Wenhao Yu](https://wyu97.github.io/) (ND), [Lingfei Wu](https://sites.google.com/a/email.wm.edu/teddy-lfwu/) (IBM), Yu Deng (IBM), Qingkai Zeng (ND), Ruchi Mahindru (IBM), Sinem Guven (IBM), [Meng Jiang](http://meng-jiang.com/) (ND).

## Installation


After cloning this repo, install dependencies using 
```
pip install -r requirements.txt
```

If you want to run with `fp16`, you need to install [Apex]( https://github.com/NVIDIA/apex.git)

## Download the dataset

The dataset can be found here: http://ibm.biz/Tech_QA

The dataset contains three files (i) `training_Q_A.json`; (ii) `dev_Q_A.json`; (iii) `training_dev_technotes.json`. 
After downloading the dataset, you should put them in the TransTD folder and set their paths to the training script.


## Training a model

In order to train a model on TechQA, use the script below. 

Note: Since TechQA is smaller dataset, it is better to start with a model that is already trained on a bigger QA dataset. Here, we start with BERT-Large trained on Squad.

```
python run_techqa.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --do_lower_case \
    --num_train_epochs 20 \
    --learning_rate 5.5e-6 \
    --do_train \
    --train_file <PATH TO training_Q_A.json> \
    --do_eval \
    --predict_file <PATH TO dev_Q_A.json> \
    --input_corpus_file <PATH TO training_dev_technotes.json> \
    --overwrite_output_dir \
    --alpha 1.0 \
    --output_dir <PATH TO OUTPUT FOLDER> \ 
    --add_doc_title_to_passage \
```

You can add the `--fp16` flag if you have apex installed.

To evaluate a model, you can run:

```
python run_techqa.py \
    --model_type bert \
    --model_name_or_path <PATH TO TRAINED MODEL FOLDER> \
    --do_lower_case \
    --do_eval \
    --predict_file <PATH TO dev_Q_A.json> \
    --input_corpus_file <PATH TO training_dev_technotes.json> \
    --overwrite_output_dir \
    --output_dir <PATH TO OUTPUT FOLDER> \ 
    --add_doc_title_to_passage 
```


## Contact

For help or issues, please submit a GitHub issue.

For direct communication, please contact Wenhao Yu (wyu1@nd.edu).


## Acknowledgement
Many thanks to the Github repository of [TechQA Baseline](https://github.com/IBM/techqa) provided by IBM. Part of our codes are modified based on their codes.

## Citation
If you find this repository useful in your research, please consider to cite our paper:

```
@inproceedings{yu2021technical,
  title={Technical Question Answering across Tasks and Domains},
  author={Yu, Wenhao and Wu, Lingfei and Deng, Yu and Zeng, Qingkai and Mahindru, Ruchi and Guven, Sinem and Jiang, Meng},
  booktitle={NAACL},
  year={2021}
}
```
