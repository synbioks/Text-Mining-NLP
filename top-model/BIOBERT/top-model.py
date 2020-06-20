#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import keras
import json
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import sys

bert_dir = "/sbksvol/gaurav/bert"



sys.path.append(bert_dir)

# In[2]:


from collections import defaultdict
from extract_features import main


# In[3]:
root_path = "/sbksvol/gaurav/BiLSTM-CRF/"
log_dir = os.path.join(root_path, "logs_sbks")
f2 = os.path.join(root_path, 'input.txt')

embedding_dir = "/sbksvol/gaurav/BiLSTM-CRF/embeddings"

### Hyperparams
MAX_LEN = 128 # maximum # of words in a sentence
EMBEDDING = 768 # dimension of word embedding vector
EPOCHS = 8
BATCH_SIZE = 64


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


# converting the tags for words to tags for tokens
# see here for an example: 
# https://docs.google.com/document/d/19WYc36YT8er45aF5-s-hxECVglufF7FruLakis67aNk/edit?usp=sharing
def extend_tags(tokens, tags, orig_tokens):
    def tag_mapper(prev_tag):
        if prev_tag == tag2idx["B"] or prev_tag == tag2idx["I"]:
            return tag2idx["I"]
        return tag2idx["O"]

    # In the comments below, original token refers to the tokens in the left column in the doc/ the input train tokens from huner
    # and the new token refers to the tokens in the right column in the doc, the output tokens from bert
    
    new_tags = [] # new_tags will store the new extended tags for the bert-tokenized sentence
    i = 0 # i is the loop variable for the new token/tag set
    tags_i = 0 # tags_i is the loop variable for the original tag set
    num_tokens = len(tokens)
    # print(tokens)
    while i < num_tokens:
        token = tokens[i]
        orig_token = orig_tokens[tags_i].lower()
        new_tags.append(tags[tags_i])
        i += 1
        
        # if the original token is not equal to the new token, it means it has been splitted by bert tokenizer
        if(token != orig_token):
            prev_tag = tags[tags_i]
            continue_tag = tag_mapper(prev_tag)

            # k stores the number of characters of the original token we have matched up. When it becomes
            # equal to the length of the orignal token (total variable), we move on to the next original token
            k = len(token) 
            total = len(orig_token)
            # so keep iterating through the new tokens until you have matched "total" number of characters
            while(k < total and i < num_tokens):
                token = tokens[i]
                # As can be seen in the second and third examples in the doc, there can be two cases of token != orig_token:
                
                # Case-1: the sub-token is preceded by a "##"
                if(len(tokens[i]) > 2 and tokens[i][:2] == "##"):
                    k += len(token) - 2 # in this case just count the # of characters in sub-token - 2
                # Case-2: the sub-token is not predeced by a "##"
                else:
                    k += len(token) # in this case just count the # of characters in sub-token

                # check if the last tag we processed was a B or I
                new_tags.append(continue_tag)
                i += 1

        tags_i += 1


    return new_tags

def transform_sentences(sentences, tags_orig, biobert_embeddings, token_vectors):
    biobert_embddgs = []
    tags_extended = []

    padder = [0.0] * EMBEDDING

    for i, sentence in enumerate(sentences):
        if(i % 100 == 0):
            print("sentence # ", i, flush=True)
        # get sentence embeddings using BERT extract_features
        biobert_embedding = biobert_embeddings[i]
        token_vector = token_vectors[i]
        # pad the sentence embedding
        embd = biobert_embedding + ([padder] * (MAX_LEN - len(biobert_embedding)))
        # extend the original tags for the WordPiece tokenized sentences
        try:
            new_tags = extend_tags(token_vector, tags_orig[i], sentence)
        except:
            print(token_vector, tags_orig[i], sentence, flush=True)
            continue
            # assert 1==0
        biobert_embddgs.append(embd)
        tags_extended.append(new_tags)
    
    return np.array(biobert_embddgs), np.array(tags_extended)

def get_data(sents, tags, embds, tokens):
    num_sents = len(sents)
    # sentences = [' '.join(sent) for sent in sents]

    X, y = transform_sentences(sents, tags, embds, tokens)
    
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])
    return X, keras.utils.to_categorical(y, num_classes=n_tags+1)

import pickle

def pickle_loader(path, split):
    
    with open(os.path.join(path, "embds_" + split + ".pickle"), "rb") as fp:
        dataX = pickle.load(fp)
    with open(os.path.join(path, "tokens_" + split + ".pickle"), "rb") as fp:
        dataY = pickle.load(fp)
    return dataX, dataY

def pickle_dumper(dataX, dataY, path, split):
    np.save(os.path.join(path, "X_" + split + ".npy"), dataX)
    np.save(os.path.join(path, "y_" + split + ".npy"), dataY)
    
    # with open(os.path.join(path, "X_" + split + ".pickle"), "wb") as fp:
    #     pickle.dump(dataX, fp)
    # with open(os.path.join(path, "y_" + split + ".pickle"), "wb") as fp:
    #     pickle.dump(dataY, fp)

print("Loading BioBERT embeddings...", flush=True)

X_tr = np.array([]).reshape(0, MAX_LEN, EMBEDDING)
y_tr = np.array([]).reshape(0, MAX_LEN, n_tags+1)
# X_te = np.array([]).reshape(0, MAX_LEN, EMBEDDING)
# y_te = np.array([]).reshape(0, MAX_LEN, n_tags+1)

# for dataset, curr_dataset in all_sents.items():
#     print(dataset, flush=True)
#     if(dataset.lower() != "bc2gm"):
#         continue
#     for split, data in curr_dataset.items():
#         (data_sents, data_tags) = data
#         print(split, flush=True)
#         path = os.path.join(embedding_dir, dataset)
#         if(not os.path.exists(path)):
#             os.mkdir(path)
#         if(split != "test"):
#             continue
#         embds, tokens = pickle_loader(path, split)
#         # X_train, y_train = get_data(data_sents, data_tags, embds, tokens)
#         X_test, y_test = get_data(data_sents, data_tags, embds, tokens)
#         pickle_dumper(X_test, y_test, path, split)
#         # embds, tokens = get_data(data_sents[:81], data_tags[:81])
        
#         print(X_test.shape, y_test.shape, flush=True)

#         X_te = np.concatenate((X_te, X_test))
#         y_te = np.concatenate((y_te, y_test))


print("Loading train data...")
for dataset in datasets:
    path = os.path.join(embedding_dir, dataset)
    X_train = np.load(os.path.join(path, "X_train.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    X_tr = np.concatenate((X_tr, X_train))
    y_tr = np.concatenate((y_tr, y_train))

from keras_contrib.utils import save_load_utils

# Design model
# ----------------------------------------------------------------------------------------
from keras.layers import Dense, Activation
from keras_contrib.layers import CRF

model = keras.Sequential()
model.add(Dense(500, input_shape=(MAX_LEN,EMBEDDING),  activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(4))
crf = CRF(len(tag2idx))
model.add(crf)
# model.add(Dense(4, input_shape=(MAX_LEN,EMBEDDING)))
# crf = CRF(len(tag2idx))
# model.add(crf)
# model.compile(optimizer="rmsprop", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
# ----------------------------------------------------------------------------------------

print("Training top model...", flush=True)


state_dict = "EPOCHS={0}_BATCH_SIZE={1}_EMBEDDING={2}_SOFTMAX_SBKS".format(EPOCHS, BATCH_SIZE, EMBEDDING)

history = model.fit(X_tr, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_split=0.1, verbose=2)
# ## Save model and history
# model.save(os.path.join(log_dir, state_dict))
# save_load_utils.save_all_weights(model,os.path.join(log_dir, state_dict))

# model = keras.models.load_model(os.path.join(log_dir, state_dict))

# model = None; save_load_utils.load_all_weights(model,os.path.join(log_dir, state_dict), include_optimizer=False)

# pred_cat = model.predict(X_tr)
# y_pred_tr = np.argmax(pred_cat, axis=-1)
# y_tr_true = np.argmax(y_tr, -1)

print("Loading test data...")

X_te = np.load(os.path.join(embedding_dir, "BC2GM", "X_test.npy"))
y_te = np.load(os.path.join(embedding_dir, "BC2GM", "y_test.npy"))

pred_cat = model.predict(X_te)
y_pred = np.argmax(pred_cat, axis=-1)
y_true = np.argmax(y_te, -1)

# with open(os.path.join(log_dir, "history_sbks.pickle"), "wb") as fp:
#     pickle.dump(history.history, fp)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def get_measures(yTrue, yPred):
    y1 = yTrue.reshape(1,-1).squeeze()
    y2 = yPred.reshape(1,-1).squeeze()

    P = precision_score(y1, y2, average=None)
    R = recall_score(y1, y2, average=None)
    F1 = f1_score(y1, y2, average=None)

    print("Precision=", flush=True)
    print(P, flush=True)
    print("Recall=", flush=True)
    print(R, flush=True)
    print("F1 score=", flush=True)
    print(F1, flush=True)


# print("Train...", flush=True)
# get_measures(y_tr_true, pred_train)

print("Test...", flush=True)
get_measures(y_true, y_pred)

# np.save("y_tr_true.npy", y_tr_true)
# np.save("pred_train.npy", pred_train)

# np.save("y_true.npy", y_true)
# np.save("y_pred.npy", y_pred)

save_load_utils.save_all_weights(model,os.path.join(log_dir, state_dict))
