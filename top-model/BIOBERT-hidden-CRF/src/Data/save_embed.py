#!/usr/bin/env python
# coding: utf-8

"""

Tag extension: create tags for newly created sub-words

MAX_LEN: 512. Max sequence length
EMBEDDING: 768
n_tags: 3
data_dir: NER/data/raw/Gene. Change to other datasets if needed

pickle_dumper
    X_train: (#sentences, MAX_LEN, EMBEDDING)
    y_train: (#sentences, n_tags+1, EMBEDDING)
    p_train: (#sentences, _) stores the position of the first sub-word of each token. 

"""

import os
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
import sys
from collections import defaultdict

bert_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/bert" 
sys.path.append(bert_dir)
embedding_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/interim/Gene/"

#############################################
MAX_LEN = 512
EMBEDDING = 768 
n_tags = 3
tag2idx = {"B" : 1, "I" : 2, "O" : 3}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}
#############################################

data_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/raw/Gene/"
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
        
        if dataset != 'miRNA': continue
        
        curr_dataset = {}
        for split in splits:
            
            if split!='test': continue
            file_path = os.path.join(data_dir, dataset, map_split(split) + ".tsv")
            sents_func, tags_func = get_sentences_helper(file_path)
            sents[split] += sents_func
            tags[split] += tags_func
            curr_dataset[split] = (sents_func, tags_func)
        
        all_sents[dataset] = curr_dataset

    return sents, tags, bc2gm_indexes, all_sents

sents, tags, bc2gm_indexes, all_sents = get_sentences()

num_train_sents = len(sents["train"])
num_devel_sents = len(sents["val"])
num_test_sents = len(sents["test"])

train_sents = sents["train"]
train_tags = tags["train"]
val_sents = sents["val"]
val_tags = tags["val"]
test_sents = sents["test"]
test_tags = tags["test"]


def extend_tags(tokens, tags, orig_tokens):
            
    def tag_mapper(prev_tag):       
        if prev_tag == tag2idx["B"] or prev_tag == tag2idx["I"]:
            return tag2idx["I"]
        return tag2idx["O"]
    
    first_token = [] # save the position of the first sub-word token of each token
    new_tags = [] 
    i = 0 
    tags_i = 0
    
    num_tokens = len(tokens)

    while i < num_tokens:
        token = tokens[i]
        orig_token = orig_tokens[tags_i].lower()
        new_tags.append(tag2idx[tags[tags_i]]) # map from O to 3 
        first_token.append(i) 
        i += 1
        if(token != orig_token):
            prev_tag = tags[tags_i]
            continue_tag = tag_mapper(prev_tag)
            
            k = len(token) 
            total = len(orig_token)
            while(k < total and i < num_tokens):
                
                token = tokens[i]
                # Case-1: the sub-token is preceded by a "##"
                if(len(tokens[i]) > 2 and tokens[i][:2] == "##"):
                    k += len(token) - 2 
                    
                # Case-2: the sub-token is not predeced by a "##"
                else:
                    # if the sub-token is unknown
                    if token=="[UNK]": 
                        i += 1
                        break
                        
                    k += len(token) 
                    
                new_tags.append(continue_tag)
                i += 1       
        tags_i += 1
    return new_tags, first_token


def transform_sentences(sentences, tags_orig, biobert_embeddings, token_vectors):
    
    biobert_embddgs = []
    tags_extended = []
    first_token_lis = []
    COUNT = 0 
    padder = [0.0] * EMBEDDING
    
    for i, sentence in enumerate(sentences):
        
        if sentence == []: continue
        biobert_embedding = biobert_embeddings[i]
        token_vector = token_vectors[i]

        # pad the sentence embedding
        embd = biobert_embedding + ([padder] * (MAX_LEN - len(biobert_embedding)))        
        new_tags, first_token = extend_tags(token_vector, tags_orig[i], sentence)

        biobert_embddgs.append(embd)
        tags_extended.append(new_tags)
        first_token_lis.append(first_token)
    
    return np.array(biobert_embddgs), np.array(tags_extended), np.array(first_token_lis)

def get_data(sents, tags, embds, tokens):
      
    num_sents = len(sents)
    sentences = [' '.join(sent) for sent in sents]
    X, y, pos = transform_sentences(sents, tags, embds, tokens)
    
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])
    
    return X, keras.utils.to_categorical(y, num_classes=n_tags+1), pos

import pickle

def pickle_loader(path, split):
    
    with open(os.path.join(path, "embds_new_" + split + ".pickle"), "rb") as fp:
        dataX = pickle.load(fp)
    with open(os.path.join(path, "tokens_new_" + split + ".pickle"), "rb") as fp:
        dataY = pickle.load(fp)
    return dataX, dataY

def pickle_dumper(dataX, dataY, dataP, path, split):
    np.save(os.path.join(path, "X_" + split + ".npy"), dataX)
    np.save(os.path.join(path, "y_" + split + ".npy"), dataY)
    np.save(os.path.join(path, "p_" + split + ".npy"), dataP)
    
    
X_tr = np.array([]).reshape(0, MAX_LEN, EMBEDDING)
y_tr = np.array([]).reshape(0, MAX_LEN, n_tags+1)

for dataset, curr_dataset in all_sents.items():

    if dataset != 'miRNA': continue
    
    for split, data in curr_dataset.items():
        
        if split != "test": continue
        
        (data_sents, data_tags) = data     
        path = os.path.join(embedding_dir, dataset)

        if(not os.path.exists(path)):
            os.mkdir(path)

        embds, tokens = pickle_loader(path, split)
        X_train, y_train, position = get_data(data_sents, data_tags, embds, tokens)  
        
        pickle_dumper(X_train, y_train, position, path, split) 
