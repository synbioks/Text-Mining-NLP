#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from collections import defaultdict

root_path = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/"

### Hyperparams
MAX_LEN = 512 # maximum # of words in a sentence
EMBEDDING = 768 # dimension of word embedding vector
EPOCHS = 4
BATCH_SIZE = 512
n_tags = 3

tag2idx = {"B" : 1, "I" : 2, "O" : 3}
tag2idx["PAD"] = 0

# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}

data_dir = "gene_data"
datasets = [x for x in os.listdir(data_dir)]
if(".DS_Store" in datasets):
    datasets.remove(".DS_Store")
if(".ipynb_checkpoints" in datasets):
    datasets.remove(".ipynb_checkpoints")
    
all_sents = {}

def get_sentences_helper(fname):
    with open(fname, "r") as fp:
        lines = [line.strip() for line in fp.readlines()]
    sentences = []
    tags = []
    sentence = []
    tag = []
    print(len(lines))
    for line in lines:
        if(len(line)):
            ent, tg = line.split("\t")
            sentence.append(ent)
            tag.append(tag2idx[tg])
        else:
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    return sentences, tags

def get_sentences():
    sents = defaultdict(list)
    tags = defaultdict(list)
    splits = ["train", "val", "test"]

    map_split = lambda x : "devel" if x =="val" else x

    bc2gm_indexes = []

    for dataset in datasets:
        
        if dataset!="Osiris":
            continue
        curr_dataset = {}
        for split in splits:
            file_path = os.path.join(data_dir, dataset, map_split(split) + ".tsv")
            sents_func, tags_func = get_sentences_helper(file_path)
            sents[split] += sents_func
            tags[split] += tags_func
            curr_dataset[split] = (sents_func, tags_func)
        
        all_sents[dataset] = curr_dataset

    return sents, tags, bc2gm_indexes, all_sents

sents, tags, bc2gm_indexes, all_sents = get_sentences()
np.save("Osiris.npy", sents)
