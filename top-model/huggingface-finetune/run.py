#!/usr/bin/env python
# coding: utf-8

"""
Architecture: NER Top Model Training setups (B) 
              Train top model first, then fine tune BioBERT+top models. Top models: 3layer-CRF, 3layer-Softmax
"""


from models.models_enum import ModelsType
from models_factory import get_model
from seqeval.metrics import f1_score, classification_report
from utils_ner import NerDataset, Split
from data_utils import convert_tsv_to_txt, get_train_test_df
import argparse
from transformers import Trainer, EarlyStoppingCallback
from transformers import TrainingArguments
from transformers.hf_argparser import HfArgumentParser
from transformers import (
    BertConfig,
    BertTokenizer
)
import transformers
import torch.nn as nn
import torch
import numpy as np
import random
import json
import gc
import os

# Important parameters:
params = {}
# params['LOWER_CASE'] = False
# params['LOAD_BEST_MODEL'] = True
# params['MAX_LEN'] = 256
# params['BATCH_SIZE'] = 32
# params['EPOCH_TOP'] = 100
# params['EPOCH_END2END'] = 100


# # All file paths have to be absolute paths
# params['WORKING_DIR'] = "sbksvol/xiang/"
# params['DATA_PATH'] = params['WORKING_DIR'] + "NER_data/"
# params['CACHE_DIR'] = params['WORKING_DIR'] + "NER_out_test"

# # Where model checkpoints are stored.
# params['OUTPUT_DIR'] = params['WORKING_DIR'] + \
#     "sbks-ucsd/top-model/huggingface-finetune/model_output_test/"
# params['TRAIN_ARGS_FILE'] = params['WORKING_DIR'] + \
#     "sbks-ucsd/top-model/huggingface-finetune/train_args_test.json"


def random_seed_set(seed_value):

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_data():
    ENT = params['entity_type']
    DATASET = params['dataset']
    data_dir = os.path.join(params['DATA_PATH'], ENT, DATASET)

    # Prepare Data
    all_data = convert_tsv_to_txt(data_dir)
    train_df, test_df, dev_df = get_train_test_df(all_data)

    # Print some data statistics
    num_train_sents = len(all_data["train"]["words"])
    num_dev_sents = len(all_data["dev"]["words"])
    num_test_sents = len(all_data["test"]["words"])
    print("num_train_sents, num_dev_sents, num_test_sents = ",
          num_train_sents, num_dev_sents, num_test_sents)

    print("First 10 words in test data:")
    print(test_df.head(10))

    # a list that has all possible labels
    labels = np.sort(train_df['labels'].unique()).tolist()
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    print("unique labels:", labels)

    return train_df, test_df, dev_df, labels, num_labels, label_map, data_dir


def prepare_config_and_tokenizer(data_dir, labels, num_labels, label_map):

    # Model Definition
    model_args = dict()
    # Path to pretrained model or model identifier from huggingface.co/models
    model_args['model_name_or_path'] = 'dmis-lab/biobert-base-cased-v1.1'
    # Where do you want to store the pretrained models downloaded from s3
    model_args['cache_dir'] = params['CACHE_DIR']
    model_args['do_basic_tokenize'] = False

    data_args = dict()
    data_args['data_dir'] = data_dir
    # "The maximum total input sequence length after tokenization. Sequences longer "
    # "than this will be truncated, sequences shorter will be padded."
    data_args['max_seq_length'] = params['MAX_LEN']
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

    tokenizer = BertTokenizer.from_pretrained(
        model_args['model_name_or_path'],
        cache_dir=model_args['cache_dir'],
        do_lower_case=params['LOWER_CASE']
    )

    return data_args, model_args, config, tokenizer


def run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map):
    # First freeze bert weights and train
    model = get_model(
        model_path=model_args["model_name_or_path"],
        cache_dir=model_args['cache_dir'],
        config=config,
        model_type=params['model_type'])

    if not params['grad_e2e']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Change from default eval mode to train mode
    model.train()

    training_args_dict = {
        'output_dir': params['OUTPUT_DIR'],
        'num_train_epochs': params['EPOCH_TOP'],
        'train_batch_size': params['BATCH_SIZE'],
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": params['LOAD_BEST_MODEL'],
        "learning_rate": params["lr"],
        "weight_decay": params["weight_decay"]
    }

    with open(params['TRAIN_ARGS_FILE'], 'w') as fp:
        json.dump(training_args_dict, fp)
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(
        json_file=params['TRAIN_ARGS_FILE'])[0]

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training
    trainOutput = trainer.train()
    trainer.save_model(params['OUTPUT_DIR'])

    if params['grad_finetune']:

        # Now reload the model from best model we have found
        # Reading from file
        print("The file is loaded from ---------------------------> ",
              params['OUTPUT_DIR']+'config.json')
        data = json.loads(open(params['OUTPUT_DIR']+'config.json', "r").read())
        top_model_path = data['_name_or_path']
        checkpoint = top_model_path.split("/")[-1]
        print("checkpoint is at ... ", checkpoint)
        print("top_model_path is at ...", params['LOAD_BEST_MODEL'])

        # Config #
        config = BertConfig.from_pretrained(
            top_model_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=model_args['cache_dir']
        )

        # Model #
        reloaded_model = get_model(
            model_path=top_model_path,
            cache_dir=model_args['cache_dir'],
            config=config,
            model_type=params['model_type'])

        # Training args #
        training_args_dict = {
            'output_dir': params['OUTPUT_DIR'],
            'num_train_epochs': params['EPOCH_END2END'],
            'train_batch_size': params['BATCH_SIZE'],
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": params['LOAD_BEST_MODEL'],
            "learning_rate": params["lr"],
            "weight_decay": params["weight_decay"]
        }

        with open(params['TRAIN_ARGS_FILE'], 'w') as fp:
            json.dump(training_args_dict, fp)
        parser = HfArgumentParser(TrainingArguments)
        training_args = parser.parse_json_file(
            json_file=params['TRAIN_ARGS_FILE'])[0]

        # Then unfreeze the bert weights and fine tune end-to-end
        model = reloaded_model
        COUNT = 1
        for param in model.base_model.parameters():
            if COUNT >= params['grad_finetune_layers']:
                param.requires_grad = True
            COUNT += 1

        model.to('cuda')

        # Set to train mode.
        model.train()

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # checkpiont is here.
        trainer.train(checkpoint)

    return trainer, model


def get_predictions_old(trainer, model, test_dataset, label_map):
    # last layer output/activation has the shape of (batch_size, seq_len,num_of_labels)
    output, label_ids, metrics = trainer.predict(test_dataset)
    preds = np.argmax(output, axis=2)
    batch_size, seq_len = preds.shape
    # preds -> list of token-level predictions shape = (batch_size, seq_len)
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def get_predictions(trainer, model, test_dataset, label_map):
    # last layer output/activation has the shape of (batch_size, seq_len,num_of_labels)
    output, label_ids, metrics = trainer.predict(test_dataset)

    if params['model_type'] == ModelsType.CRF:
        batch_size, seq_len, num_labels = output.shape
        output = torch.tensor(output).to('cuda')

        all_attention_masks = []
        for sample in test_dataset:
            all_attention_masks.append(sample.attention_mask)
        all_attention_masks = torch.tensor(all_attention_masks).to('cuda')

        # get the best tag sequences using CRF's Viterbi decode algorithm
        preds = model.crf.decode(output, all_attention_masks.type(torch.uint8))
    else:
        preds = np.argmax(output, axis=2)
        batch_size, seq_len = preds.shape

    # preds -> list of token-level predictions shape = (batch_size, seq_len)
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(len(preds[i])):
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
    test_labels = test_labels[:len(preds_list)]
    test_labels_new = []
    preds_list_new = []

    for i, x in enumerate(test_labels):
        if len(x) == len(preds_list[i]):
            test_labels_new.append(x)
            preds_list_new.append(preds_list[i])
        else:
            print("ABORT")

    try:    
        print("F1-score: {:.1%}".format(f1_score(test_labels_new, preds_list_new)))
        print(classification_report(test_labels_new, preds_list_new, digits=3))
    except Exception as e:
        print(f'Error - {e}')


def main(_params):
    global params
    params = _params

    train_df, test_df, dev_df, labels, num_labels, label_map, data_dir = prepare_data()

    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(
        data_dir, labels, num_labels, label_map)

    # ## Create Dataset Objects

    train_dataset = NerDataset(
        data_dir=data_args['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args['max_seq_length'],
        overwrite_cache=data_args['overwrite_cache'],  # True
        mode=Split.train,
        data_size=params["data_size"])

    eval_dataset = NerDataset(
        data_dir=data_args['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args['max_seq_length'],
        overwrite_cache=data_args['overwrite_cache'],
        mode=Split.dev,
        data_size=params["data_size"])

    print(train_dataset.__len__(), eval_dataset.__len__())

    # Train top-model using the Trainer API
    trainer, model = run_train(train_dataset, eval_dataset,
                               config, model_args, labels, num_labels, label_map)

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
        mode=Split.test,
        data_size=params["data_size"])

    run_test(trainer, model, train_dataset, train_df, label_map)
    run_test(trainer, model, eval_dataset, dev_df, label_map)
    run_test(trainer, model, test_dataset, test_df, label_map)


# if __name__ == "__main__":
#     main()
