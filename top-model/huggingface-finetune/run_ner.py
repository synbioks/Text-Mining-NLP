#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np

# python_site_packages_path = "/root/anaconda3/lib/python3.7/site-packages"
import sys
# print(sys.path)
# if not python_site_packages_path in sys.path:
#     sys.path.append(python_site_packages_path)


import torch
import torch.nn as nn
import transformers
from transformers import (
    BertConfig,
    BertTokenizer
)
from transformers.hf_argparser import HfArgumentParser
from transformers import TrainingArguments
from transformers import Trainer
import json

parser = argparse.ArgumentParser()
parser.add_argument("--seed_value", default=42)
parser.add_argument("--entity_type", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

data_path = "/sbksvol/gaurav/NER_data/"

from data_utils import convert_tsv_to_txt, get_train_test_df
from utils_ner import NerDataset, Split
from models import BertNERTopModel

def random_seed_set(seed_value):

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    
def prepare_data():
    ENT = args.entity_type
    DATASET = args.dataset
    data_dir = os.path.join(data_path, ENT, DATASET)

    # ## Prepare Data

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
    # saved_model_path
    # saved_model_path
    # pytorch_dump_path
    # 'dmis-lab/biobert-base-cased-v1.1'

    # Where do you want to store the pretrained models downloaded from s3
    model_args['cache_dir'] = "/sbksvol/gaurav/NER_out/"

    # we skip basic white-space tokenization by passing do_basic_tokenize = False to the tokenizer
    model_args['do_basic_tokenize'] = False


    data_args = dict()

    data_args['data_dir'] = data_dir

    # "The maximum total input sequence length after tokenization. Sequences longer "
    # "than this will be truncated, sequences shorter will be padded."
    data_args['max_seq_length'] = 256

    # Overwrite the cached training and evaluation sets
    # this means the model does not have to tokenize/preprocess and cache the data each time it's called
    # this can be made different for each NerDataset (training NerDataset, testing NerDataset)
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
        cache_dir=model_args['cache_dir']
    #     ,do_basic_tokenize = model_args['do_basic_tokenize']
    )
    
    return data_args, model_args, config, tokenizer

def run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map):
    top_model = {"name": args.model, 
                 "hidden_units_list": [500, 250, 125], 
                 "activations_list": ["none", "none", "none", "none"]
                }


    # ### First freeze bert weights and train

    model = BertNERTopModel.from_pretrained(
        model_args['model_name_or_path'],
        config=config,
        cache_dir=model_args['cache_dir'],
        top_model=top_model
    )

    ## base_model -> bert (excluding the classification layer)
    for param in model.base_model.parameters():
        param.requires_grad = False


    model.train()
    
    training_args_dict = {
        'output_dir' : "model_output/",
        'num_train_epochs' : 20,
        'train_batch_size': 32,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch"
    #     ,
    #     "load_best_model_at_end": True
    }
    
    # ### Create Trainer
    with open('training_args.json', 'w') as fp:
        json.dump(training_args_dict, fp)
    
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file="training_args.json")[0]


    # ## Train

    # Initialize the Trainer
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset
    )

    trainOutput = trainer.train()

    # ## Now reload the model from saved checkpoint

    num_steps = trainOutput.global_step # 17880
    checkpoint = f"checkpoint-{num_steps}"
    top_model_path = f"{training_args_dict['output_dir']}/{checkpoint}" 

    #### Config ####
    config = BertConfig.from_pretrained(
        top_model_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args['cache_dir']
    )

    #### Model ####

    reloaded_model = BertNERTopModel.from_pretrained(
        top_model_path,
        config=config,
        cache_dir=model_args['cache_dir'],
        top_model=top_model
    )


    #### Training args ####
    training_args_dict = {
        'output_dir' : "model_output",
        'num_train_epochs' : 5,
        'train_batch_size': 32,
        'seed':int(args.seed_value),
        "evaluation_strategy": "epoch"
#         ,"load_best_model_at_end": True
    }

    with open('training_args.json', 'w') as fp:
        json.dump(training_args_dict, fp)

    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file="training_args.json")[0]


    # ## Then unfreeze the bert weights and train end-to-end

    model = reloaded_model

    for param in model.base_model.parameters():
        param.requires_grad = True


    model.to('cuda')
    model.train()
    
#     print("model")


    # Initialize our Trainer
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset
    )

    # Begin training from the latest checkpoint
    trainer.train(checkpoint)
    # optionally, save the trained model
    # trainer.save_model()
    
    return trainer, model

def get_predictions(trainer, model, test_dataset, label_map, model_name):
    # last layer output/activation has the shape of (batch_size, seq_len,num_of_labels)
    output, label_ids, metrics = trainer.predict(test_dataset)
    
    
    if "softmax" in model_name:
        preds = np.argmax(output, axis=2)
        batch_size, seq_len = preds.shape
    elif "crf" in model_name:
        batch_size, seq_len, num_labels = output.shape
        output = torch.tensor(output).to('cuda')
        
        all_attention_masks = []
        for sample in test_dataset:
            all_attention_masks.append(sample.attention_mask)
        all_attention_masks = torch.tensor(all_attention_masks).to('cuda')

        # get the best tag sequences using CRF's Viterbi decode algorithm
        preds = model.crf.decode(output, all_attention_masks.type(torch.uint8))

    # preds -> list of token-level predictions shape = (batch_size, seq_len)
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            # ignore pad_tokens
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list
                

def run_test(trainer, model, test_dataset, test_df, label_map):
    preds_list = get_predictions(trainer, model, test_dataset, label_map, model_name=args.model)


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


    # ## Get entity level scores

    # In[49]:


    from seqeval.metrics import f1_score, classification_report
    print("F1-score: {:.1%}".format(f1_score(test_labels_new, preds_list_new)))
    print(classification_report(test_labels_new, preds_list_new))
    

def main():
    random_seed_set(int(args.seed_value))

    train_df, test_df, labels, num_labels, label_map, data_dir = prepare_data()
    
    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(data_dir, labels, num_labels, label_map)

    # ## Create Dataset Objects

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


    # ## Train top-model using the Trainer API
    trainer, model = run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map)

    # ## Clean-up

    import gc
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