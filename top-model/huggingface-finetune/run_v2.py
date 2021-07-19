#!/usr/bin/env python
# coding: utf-8

"""
Architecture: NER Top Model Training setups (B)
              Train top model first, then fine tune BioBERT+top models. Top models: 3layer-CRF, 3layer-Softmax
"""
from transformers.trainer_utils import set_seed
from models.models_enum import ModelsType
import os
import gc
import json
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizer
)
from transformers.hf_argparser import HfArgumentParser
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
from utils import LogCallback, plot_loss_log

from data_utils import convert_tsv_to_txt, get_train_test_df
from utils_ner import NerDataset, Split
from models_factory import get_model
from seqeval.metrics import f1_score, classification_report
params = {}
# Important parameters:
# params["LOWER_CASE"] = False
# params["LOAD_BEST_MODEL"] = True
# params["MAX_LEN"] = 256
# params["BATCH_SIZE"] = 64
# params["EPOCH_TOP"] = 100
# params["EPOCH_END2END"] = 100

# # All file paths have to be absolute paths

# params["DATA_PATH"] = "sbksvol/xiang/" + "NER_data/"


# parser = argparse.ArgumentParser()
# parser.add_argument("--seed_value", default=42)
# parser.add_argument("--set_seed", required=True)
# parser.add_argument("--entity_type", required=True)
# parser.add_argument("--dataset", required=True)
# parser.add_argument("--exp_name", default="base")
# args = parser.parse_args()

# params["EXP_NAME"] = args.exp_name
# params["WORKING_DIR"] = "sbksvol/nikhil/" + params["EXP_NAME"] + "/"
# params["CACHE_DIR"] = params["WORKING_DIR"] + "NER_out_test/"

# # Where model checkpoints are stored.
# params["OUTPUT_DIR"] = params["WORKING_DIR"] + "model_output_test/"
# params["TRAIN_ARGS_FILE"] = params["WORKING_DIR"] + "train_args_test.json"


def random_seed_set(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    #os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed_value)

def process_entity(tokenizer, train_df,add_to_vocab=True):
    temp_df = train_df.copy()
    train_df.loc[temp_df['labels']=='B',['words']] = temp_df[temp_df['labels']=='B']['words'].apply(lambda x:x.replace('-',''))
    print('Unique Entities - ', train_df[train_df['labels']=='B']['words'].unique())
    if add_to_vocab:
        print('Adding entities to vocab')
        tokenizer.add_tokens(list(train_df[train_df['labels']=='B']['words'].unique()))
            
def prepare_data():
    ENT = params["entity_type"]
    DATASET = params["dataset"]
    data_dir = os.path.join(params["DATA_PATH"], ENT, DATASET)

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
    model_args['cache_dir'] = params["CACHE_DIR"]
    model_args['do_basic_tokenize'] = False

    data_args = dict()
    data_args['data_dir'] = data_dir
    # "The maximum total input sequence length after tokenization. Sequences longer "
    # "than this will be truncated, sequences shorter will be padded."
    data_args['max_seq_length'] = params["MAX_LEN"]
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
        do_lower_case=params["LOWER_CASE"]
    )

    return data_args, model_args, config, tokenizer


def run_train(train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map,tokenizer):
    # First freeze bert weights and train
    model = get_model(
        model_path=model_args["model_name_or_path"],
        cache_dir=model_args['cache_dir'],
        config=config,
        model_type=params['model_type'])

    if not params['grad_e2e']:
        for param in model.base_model.parameters():
            param.requires_grad = False
    if 'add_vocab' in params.keys():
        model.resize_token_embeddings(len(tokenizer))
        for param in model.bert.embeddings.parameters():
            param.requires_grad = True

    # Change from default eval mode to train mode
    model.train()
    print(model)

    training_args_dict = {
        'output_dir': params["OUTPUT_DIR"],
        'num_train_epochs': params["EPOCH_TOP"],
        'train_batch_size': params["BATCH_SIZE"],
        "save_strategy": "epoch",
        "evaluation_strategy": "steps",
        "eval_steps": max(10,train_dataset.__len__()//params["BATCH_SIZE"]),
        "logging_steps":max(10,train_dataset.__len__()//params["BATCH_SIZE"]),
        "do_train": True,
        "load_best_model_at_end": params["LOAD_BEST_MODEL"],
        "learning_rate": params["lr"],
        "weight_decay": params["weight_decay"],
        "save_total_limit": 2
    }
    print(training_args_dict)
    with open(params["TRAIN_ARGS_FILE"], 'w') as fp:
        json.dump(training_args_dict, fp)
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(
        json_file=params["TRAIN_ARGS_FILE"])[0]

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, callbacks=[
            EarlyStoppingCallback(early_stopping_patience=params["patience"]),
            LogCallback(params["OUTPUT_DIR"]+"/train_log.json")]
    )

    # Start training
    trainOutput = trainer.train()
    trainer.save_model(params["OUTPUT_DIR"])
    plot_loss_log(params["OUTPUT_DIR"]+"/train_log.json")
    best_model = trainer.state.best_model_checkpoint

    if params['grad_finetune']:

        # Now reload the model from best model we have found
        # Reading from file
        print("The file is loaded from ---------------------------> ",
              params["OUTPUT_DIR"] + 'config.json')
        data = json.loads(
            open(params["OUTPUT_DIR"] + 'config.json', "r").read())
        top_model_path = best_model
        checkpoint = top_model_path.split("/")[-1]
        print("checkpoint is at ... ", checkpoint)
        print("top_model_path is at ...", top_model_path)

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
            model_path=top_model_path+"/",
            cache_dir=model_args['cache_dir'],
            config=None,
            model_type=params['model_type'])
        print("Reloaded",reloaded_model.bert.embeddings)

        # Training args #
        training_args_dict = {
            'output_dir': params["OUTPUT_DIR"],
            'num_train_epochs': params["EPOCH_TOP"] + params["EPOCH_END2END"],
            'train_batch_size': params["BATCH_SIZE"],
            "evaluation_strategy": "steps",
            "eval_steps": max(10,train_dataset.__len__()//params["BATCH_SIZE"]),
            "logging_steps":max(10,train_dataset.__len__()//params["BATCH_SIZE"]),
            "do_train": True,
            "load_best_model_at_end": params["LOAD_BEST_MODEL"],
            "save_total_limit": 2,
            "learning_rate": params["lr_finetune"],
            "weight_decay": params["wd_finetune"] if "wd_finetune" in params.keys() else 0,
            "ignore_data_skip": True
        }

        with open(params["TRAIN_ARGS_FILE"], 'w') as fp:
            json.dump(training_args_dict, fp)
        parser = HfArgumentParser(TrainingArguments)
        training_args = parser.parse_json_file(
            json_file=params["TRAIN_ARGS_FILE"])[0]

        # Then unfreeze the bert weights and fine tune end-to-end
        model = reloaded_model
        if params['grad_finetune_layers'] != -1:
            for param in model.base_model.parameters():
                param.requires_grad = False

            for i in range(len(model.bert.encoder.layer) - params['grad_finetune_layers'], len(model.bert.encoder.layer)):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
        model.to('cuda')

        # Set to train mode.
        model.train()

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=params["patience"]),
            LogCallback(params["OUTPUT_DIR"]+"/train_finetune_log.json")]
        )

        # checkpiont is here.
        trainer.train()
        plot_loss_log(params["OUTPUT_DIR"]+"/train_finetune_log.json")
    return trainer, model


def get_predictions2(trainer, model, test_dataset, label_map):
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

    if params['model_type'] == ModelsType.CRF or params['model_type'] == ModelsType.FCN_CRF:
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
        def tupple_function(x): return [(w, t) for w, t in zip(x["words"].values.tolist(),
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

    print("F1-score: {:.1%}".format(f1_score(test_labels_new, preds_list_new)))
    print(classification_report(test_labels_new, preds_list_new, digits=3))


def main(_params):
    global params
    params = _params
    '''
    params['seed_value'] = args.seed_value
    params['set_seed'] = args.set_seed
    '''
    if params['set_seed']:
        random_seed_set(params['seed_value'])

    train_df, test_df, dev_df, labels, num_labels, label_map, data_dir = prepare_data()

    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(
        data_dir, labels, num_labels, label_map)
    
    if 'add_vocab' in params.keys():
        process_entity(tokenizer,train_df)
        process_entity(tokenizer,dev_df)
        process_entity(tokenizer,test_df)

    # ## Create Dataset Objects

    train_dataset = NerDataset(
        data_dir=data_args['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args['max_seq_length'],
        overwrite_cache=data_args['overwrite_cache'],  # True
        mode=Split.train, data_size=params["data_size"])

    eval_dataset = NerDataset(
        data_dir=data_args['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args['max_seq_length'],
        overwrite_cache=data_args['overwrite_cache'],
        mode=Split.dev, data_size=100)

    print(train_dataset.__len__(), eval_dataset.__len__())

    # Train top-model using the Trainer API
    trainer, model = run_train(
        train_dataset, eval_dataset, config, model_args, labels, num_labels, label_map,tokenizer)

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
        mode=Split.test, data_size=100)
    run_test(trainer, model, train_dataset, train_df, label_map)
    run_test(trainer, model, eval_dataset, dev_df, label_map)
    run_test(trainer, model, test_dataset, test_df, label_map)


# if __name__ == "__main__":
#     main()
