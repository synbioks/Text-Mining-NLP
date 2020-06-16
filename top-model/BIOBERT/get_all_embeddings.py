#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# import tensorflow as tf
# import keras
import json
import pandas as pd
import numpy as np
# from keras.preprocessing.sequence import pad_sequences
import sys

bert_dir = "/sbksvol/gaurav/bert"

sys.path.append(bert_dir)

# In[2]:


from collections import defaultdict
from extract_features import main


# In[3]:
root_path = "/sbksvol/gaurav/BiLSTM-CRF/"
f2 = os.path.join(root_path, 'input.txt')

embedding_dir = "/sbksvol/gaurav/BiLSTM-CRF/embeddings"

### Hyperparams
MAX_LEN = 128 # maximum # of words in a sentence
EMBEDDING = 768 # dimension of word embedding vector
EPOCHS = 4
BATCH_SIZE = 512


# In[9]:

n_tags = 3
###################################################################################
# Vocabulary Key:Label/Tag -> Value:tag_index
# The first entry is reserved for PAD
tag2idx = {"B" : 1, "I" : 2, "O" : 3}
tag2idx["PAD"] = 0

# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}
###################################################################################


# In[10]:


data_dir = "data"
# log_dir = "logs"

# In[130]:

datasets = [x for x in os.listdir(data_dir)]

if(".DS_Store" in datasets):
    datasets.remove(".DS_Store")

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



    # datasets.sort()

    splits = ["train", "val", "test"]

    map_split = lambda x : "devel" if x =="val" else x
    
    bc2gm_indexes = []

    for dataset in datasets:
        print(dataset, flush=True)
        if(dataset.lower() == "bc2gm"):
#             print(len(sents["train"]), len(sents["val"]), len(sents["test"]))
            bc2gm_indexes.append((len(sents["train"]), len(sents["val"]), len(sents["test"])))
        curr_dataset = {}
        for split in splits:
            file_path = os.path.join(data_dir, dataset, map_split(split) + ".tsv")
            sents_func, tags_func = get_sentences_helper(file_path)

            sents[split] += sents_func
            tags[split] += tags_func
            curr_dataset[split] = (sents_func, tags_func)

        if(dataset.lower() == "bc2gm"):
            bc2gm_indexes.append((len(sents["train"]), len(sents["val"]), len(sents["test"])))
        
        all_sents[dataset] = curr_dataset

    return sents, tags, bc2gm_indexes, all_sents


# In[131]:


sents, tags, bc2gm_indexes, all_sents = get_sentences()


# In[132]:


num_train_sents = len(sents["train"])
num_devel_sents = len(sents["val"])
num_test_sents = len(sents["test"])

print(num_train_sents, num_devel_sents, num_test_sents, flush=True)


# In[133]:


train_sents = sents["train"]
train_tags = tags["train"]
val_sents = sents["val"]
val_tags = tags["val"]
test_sents = sents["test"]
test_tags = tags["test"]

def get_bio_bert_embedding_helper(d):
    # get the embedding of each token
    sentence_vector = []
    token_vector = []
    for i in range (1, len(d['features'])-1):
        sentence_vector.append(d['features'][i]['layers'][0]['values'])
        token_vector.append(d['features'][i]['token'])

    number_of_tokens = len(d['features']) - 2
    
    # normalize the token embeddings by the # of tokens
    sentence_vector = [[element/number_of_tokens for element in elem] for elem in sentence_vector]
    # print("token_vector=", token_vector)

    return sentence_vector, token_vector


def get_bio_bert_embedding():
    biobert_data = main()
    data = [json.loads(line) for line in biobert_data]

    sentence_vectors = []
    token_vectors = []
    for data_point in data:
        sentence_vector, token_vector = get_bio_bert_embedding_helper(data_point)
        sentence_vectors.append(sentence_vector)
        token_vectors.append(token_vector)

    return sentence_vectors, token_vectors


        

def transform_sentences(sentences, tags_orig):
    """ Get BioBERT embedding and pad with values as necessasry to get the feature encoding
    of sentences
        Input:
            input_file -> one sentence on every line
    """
    
    biobert_embddgs = []
    tags_extended = []

    padder = [0.0] * EMBEDDING

    biobert_embeddings, token_vectors = get_bio_bert_embedding()

    return biobert_embeddings, token_vectors

# create the input file
def create_input_file(sentences):
    with open(f2, "w") as f_in:
        for sentence in sentences:
            f_in.write(sentence+"\n")


def get_data(sents, tags):
    embds = []; tokens = []
    num_sents = len(sents)
    sentences = [' '.join(sent) for sent in sents]
    for sent_no in range(0,num_sents,BATCH_SIZE):
        print("Batch # ", sent_no//BATCH_SIZE)
        batch_sents = sentences[sent_no:sent_no+BATCH_SIZE]
        batch_tags = tags[sent_no:sent_no+BATCH_SIZE]
        create_input_file(batch_sents)
        X, y = transform_sentences(batch_sents, batch_tags)
        embds += X
        tokens += y

    
    # print("Creating input file...", flush=True)
    
    # print("Transforming sentences...", flush=True)
    
    # y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])
    # return X, keras.utils.to_categorical(y, num_classes=n_tags+1)
    return embds, tokens

import pickle

def pickle_dumper(dataX, dataY, path, split):
    
    with open(os.path.join(path, "embds_" + split + ".pickle"), "wb") as fp:
        pickle.dump(dataX, fp)
    with open(os.path.join(path, "tokens_" + split + ".pickle"), "wb") as fp:
        pickle.dump(dataY, fp)

print("Computing BioBERT embeddings...", flush=True)

for dataset, curr_dataset in all_sents.items():
    print(dataset, flush=True)
    if(dataset.lower() != "bc2gm"):
        continue
    for split, data in curr_dataset.items():
        (data_sents, data_tags) = data
        print(split, flush=True)
        path = os.path.join(embedding_dir, dataset)
        # if(not os.path.exists(path)):
        #     os.mkdir(path)
        if(split != "test"):
            continue
        # X_train, y_train = get_data(data_sents, data_tags)
        # embds, tokens = get_data(data_sents[:81], data_tags[:81])
        embds, tokens = get_data(data_sents, data_tags)
        # pickle_dumper(X_train, y_train, path, split)
        pickle_dumper(embds, tokens, path, split)
        # print(X_train.shape, y_train.shape, flush=True)
        print(len(embds), len(tokens))

