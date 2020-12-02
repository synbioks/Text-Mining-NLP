#!/usr/bin/env python
# coding: utf-8

"""
MAX_SEQ_LEN = 512
EMBEDDING = 768
BASELINE = False, True if applying softmax or other baseline.
"""

import os
import keras
import numpy as np
import sys
from keras.layers import Dense, Activation, Masking
from metrics import get_measures
embedding_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/interim/Gene/"
model_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/models/"
root_path = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/src/models/"
crf_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/keras-contrib/keras_contrib/layers" 
sys.path.append(crf_dir)
from crf import CRF

MAX_SEQ_LEN = 512
EMBEDDING = 768
BASELINE = False

if BASELINE:
    model = keras.Sequential()
    model.add(Masking(input_shape=(MAX_SEQ_LEN,EMBEDDING)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.load_weights(model_dir+"Ety=Gene_512_ECH=6_BCH=32_LR=0.01_EMBD=768_SoftMax.h5")
    model.summary()
else:
    model = keras.Sequential()
    model.add(Masking(input_shape=(MAX_SEQ_LEN,EMBEDDING)))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    crf = CRF(4)
    model.add(crf)
    # Most recent version: 10/26
    model.load_weights(model_dir+"Ety=Gene_512_ECH=4_BCH=32_LR=0.01_EMBD=768_3_layer_CRF.h5")
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

# Most recent: 10/27
X_te = np.load(os.path.join(embedding_dir, "osiris", "X_new_test.npy"))
y_te = np.load(os.path.join(embedding_dir, "osiris", "y_new_test.npy"))
test_pos = np.load(os.path.join(embedding_dir, "osiris", "p_new_test.npy"),allow_pickle=True)

pred_cat = model.predict(X_te)
y_pred = np.argmax(pred_cat, axis=-1)
y_true = np.argmax(y_te, -1)

print("subword: ")
get_measures(y_pred, y_true)

# token-level output 
res_true = []
res_pred = []
for i in range(len(y_pred)):
    res_true.append(y_true[i][test_pos[i]])
    res_pred.append(y_pred[i][test_pos[i]])
res_true = np.array(res_true)
res_pred = np.array(res_pred)

# np.save(os.path.join(model_dir, "fsu_pred_softmax.npy"), res_pred)
# np.save(os.path.join(model_dir, "fsu_true_softmax.npy"), res_true)
