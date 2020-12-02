#!/usr/bin/env python
# coding: utf-8

"""

FLAG: True when generating text files as input to BioBERT. False when computing embeddings 
EMBEDDING: 768.
n_tags: 3. IOB tagging
data_dir: dataset in NER/data/raw

Note: When computing embeddings, check which text file is used as input to BioBERT. 
      Also, change the file path in pickle_dumper

"""

import os
import json
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

data_path = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/raw/"
FLAG = True 
n_tags = 3

tag2idx = {"B" : 1, "I" : 2, "O" : 3}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

data_dir = "Chemicals"
datasets = [x for x in os.listdir(data_path+data_dir)]
splits = ["train", "val", "test"]

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
    map_split = lambda x : "dev" if x =="val" else x
    for dataset in datasets:
                    
        curr_dataset = {}
        for split in splits:
            file_path = os.path.join(data_path+data_dir, dataset, map_split(split) + ".tsv")
            
            sents_func, tags_func = get_sentences_helper(file_path)
            sents[split] += sents_func
            tags[split] += tags_func
            curr_dataset[split] = (sents_func, tags_func)
        all_sents[dataset] = curr_dataset    

    return sents, tags, all_sents

sents, tags, all_sents = get_sentences()

if FLAG: 
    for dataset in datasets:
        for split in splits:
            text = all_sents[dataset][split][0]
            save_txt = data_path + data_dir + "/" + dataset + "/" + split + ".txt"
            with open(save_txt, "w") as text_file:
                for line in text:
                    if line: 
                        text_file.write(" ".join(line) + "\n")          
    sys.exit()

# -------------------------------------------------------------------------------------------                    

embedding_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/interim/"
bert_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/bert" 
sys.path.append(bert_dir)
from extract_features import main
import pickle
def get_bio_bert_embedding_helper(d):

    sentence_vector = []
    token_vector = []
    
    # skip CLS and SEP 
    for i in range (1, len(d['features'])-1):
        sentence_vector.append(d['features'][i]['layers'][0]['values'])
        token_vector.append(d['features'][i]['token']) 
    number_of_tokens = len(d['features']) - 2 # CLS, SEP
    
    # normalize the embeddings
    sentence_vector = [[element/number_of_tokens for element in elem] for elem in sentence_vector]

    return sentence_vector, token_vector


def pickle_dumper(dataX, dataY, split):
        
    # Change the path each time
    path = embedding_dir + "Chemicals/biosemantics/"
    with open(os.path.join(path, "embds_" + split + ".pickle"), "wb") as fp:
        pickle.dump(dataX, fp)
    with open(os.path.join(path, "tokens_" + split + ".pickle"), "wb") as fp:
        pickle.dump(dataY, fp)
    
biobert_data = main() # use large amount of memory

data = [json.loads(line) for line in biobert_data]

sentence_vectors = []
token_vectors = []
for data_point in data:
    sent_vec, token_vec = get_bio_bert_embedding_helper(data_point)
    sentence_vectors.append(sent_vec)
    token_vectors.append(token_vec)
    
pickle_dumper(sentence_vectors, token_vectors, "train")           
    