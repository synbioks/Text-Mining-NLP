#!/usr/bin/env python
# coding: utf-8

"""
Architecture: NER Top Model Training setups (C) 
              End-to-end training. Top models:  softmax
"""

# Important parameters: 
LOWER_CASE = False
LOAD_BEST_MODEL = True
MAX_LEN = 256
EPOCH_END2END = 100
BATCH_SIZE = 32

# All file paths have to be absolute paths #
WORDING_DIR = "sbksvol/xiang/"
DATA_PATH = WORDING_DIR + "NER_data/"
CACHE_DIR = WORDING_DIR + "NER_out_end/"

# Where model checkpoints are stored. 
OUTPUT_DIR = WORDING_DIR + "NER_src/model_output_end/"
TRAIN_ARGS_FILE = WORDING_DIR + "NER_src/train_args_end.json"

import os
import gc
import json
import random
import numpy as np

import torch
import torch.nn as nn
import transformers
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification
)
from transformers.hf_argparser import HfArgumentParser
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed_value", default=42)
parser.add_argument("--set_seed", required=True)
parser.add_argument("--entity_type", required=True)
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

from data_utils import convert_tsv_to_txt, get_train_test_df
from utils_ner import NerDataset, Split

def random_seed_set(seed_value):

    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def prepare_data():
    
    entity = args.entity_type
    dataset = args.dataset
    data_dir = os.path.join(DATA_PATH, entity, dataset)

    ### Prepare Data
    all_data = convert_tsv_to_txt(data_dir)
    train_df, test_df = get_train_test_df(all_data)
    
    # Print some data statistics
    num_train_sents = len(all_data["train"]["words"])
    num_dev_sents = len(all_data["dev"]["words"])
    num_test_sents = len(all_data["test"]["words"])
    print("num_train_sents, num_dev_sents, num_test_sents = ", num_train_sents, num_dev_sents, num_test_sents)


    print("First 10 words in test data:")
    print(test_df.head(10))

    # a list that has all possible labels 
    labels = np.sort(train_df['labels'].unique()).tolist()
    label_map =  {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    print("unique labels:", labels)
    
    return train_df, test_df, labels, num_labels, label_map, data_dir
    
    
def prepare_config_and_tokenizer(data_dir, labels, num_labels, label_map):
    # ## Model Definition

    model_args = dict()

    # Path to pretrained model or model identifier from huggingface.co/models
    model_args['model_name_or_path'] = 'dmis-lab/biobert-base-cased-v1.1'
    model_args['cache_dir'] = CACHE_DIR
    model_args['do_basic_tokenize'] = False

    data_args = dict()
    data_args['data_dir'] = data_dir
    data_args['max_seq_length'] = MAX_LEN
    data_args['overwrite_cache'] = True
    
    config = BertConfig.from_pretrained(
        model_args['model_name_or_path'],
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args['cache_dir']
    )

    # we skip basic white-space tokenization by passing do_basic_tokenize = False to the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_args['model_name_or_path'],
        cache_dir=model_args['cache_dir'],
        do_lower_case = LOWER_CASE
    )
    
    return data_args, model_args, config, tokenizer

def run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map):

    model = BertForTokenClassification.from_pretrained(
    model_args['model_name_or_path'],
    config=config,
    cache_dir=model_args['cache_dir']
    )
    
    # train end-to-end
    for param in model.base_model.parameters():
        param.requires_grad = True

    model.to('cuda')
    model.train()
        
    # Training args #
    training_args_dict = {
        'output_dir' : OUTPUT_DIR,
        'num_train_epochs' : EPOCH_END2END,
        'train_batch_size': BATCH_SIZE,
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": LOAD_BEST_MODEL
    }
    
    # Create Trainer
    with open(TRAIN_ARGS_FILE, 'w') as fp:
        json.dump(training_args_dict, fp)
    
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file=TRAIN_ARGS_FILE)[0]

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
 
    # Initialize the Trainer
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
    )

    trainOutput = trainer.train()

    return trainer, model

def get_predictions(trainer, model, test_dataset, label_map):
    # last layer output/activation has the shape of (batch_size, seq_len,num_of_labels)
    output, label_ids, metrics = trainer.predict(test_dataset)
    preds = np.argmax(output, axis=2)
    batch_size, seq_len = preds.shape

    # preds -> list of token-level predictions shape = (batch_size, seq_len)
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            # ignore pad_tokens
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list
                

def run_test(trainer, model, test_dataset, test_df, label_map):
    preds_list = get_predictions(trainer, model, test_dataset, label_map)

    def sentences_combiner(df):
        # 'words' and 'labels' are the column names in the CSV file
        tupple_function = lambda x: [(w, t) for w, t in zip(x["words"].values.tolist(),
                                                          x["labels"].values.tolist())]
        grouped = df.groupby("sentence_id").apply(tupple_function)
        return [s for s in grouped]

    testing_sentences = sentences_combiner(test_df)
    test_labels = [[w[1] for w in s] for s in testing_sentences]
    test_tokens = [[w[0] for w in s] for s in testing_sentences]

    
    # make sure all test and pred sentences have the same length
    # for some tokenization reason cellfinder(Cellline) had a problem with 3 test sentences
    test_labels_new = []
    preds_list_new = []

    for i, x in enumerate(test_labels):
        if len(x) == len(preds_list[i]):
            test_labels_new.append(x)
            preds_list_new.append(preds_list[i])
        else:
            print("ABORT")

    from seqeval.metrics import f1_score, classification_report
    print("F1-score: {:.1%}".format(f1_score(test_labels_new, preds_list_new)))
    print(classification_report(test_labels_new, preds_list_new))

def main():
    
    # If args.dataset == True, set random seed.
    if args.set_seed == "Yes":
        print("note that random seed is set to -------> ", args.seed_value)
        random_seed_set(int(args.seed_value))

    train_df, test_df, labels, num_labels, label_map, data_dir = prepare_data()
    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(data_dir, labels, num_labels, label_map)

    ### Create Dataset Objects

    train_dataset = NerDataset(
      data_dir=data_args['data_dir'],
      tokenizer=tokenizer,
      labels=labels,
      model_type=config.model_type,
      max_seq_length=data_args['max_seq_length'],
      overwrite_cache=data_args['overwrite_cache'], # True
      mode=Split.train)

    eval_dataset = NerDataset(
      data_dir=data_args['data_dir'],
      tokenizer=tokenizer,
      labels=labels,
      model_type=config.model_type,
      max_seq_length=data_args['max_seq_length'],
      overwrite_cache=data_args['overwrite_cache'],
      mode=Split.dev)
    
    print(train_dataset.__len__(), eval_dataset.__len__())

    # Train end-to-end using the Trainer API
    trainer, model = run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map)

    gc.collect()
    torch.cuda.empty_cache()
    
    # ## Prepare test data, run trainer over test data and print metrics
    # we can pass overwrite_cache as True since we might like to make new predictions by just changing test.txt 
    test_dataset = NerDataset(
      data_dir=data_args['data_dir'],
      tokenizer=tokenizer,
      labels=labels,
      model_type=config.model_type,
      max_seq_length=data_args['max_seq_length'],
      overwrite_cache=True,
      mode=Split.test)

    run_test(trainer, model, test_dataset, test_df, label_map)

if __name__ == "__main__":
    main()
    