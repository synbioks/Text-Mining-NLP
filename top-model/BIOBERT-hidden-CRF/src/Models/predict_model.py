#!/usr/bin/env python
# coding: utf-8

import os
import keras
import numpy as np
from keras.layers import Dense, Activation, Masking
from metrics import get_measures
embedding_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/gene_embedding/"
model_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/models/"

MAX_SEQ_LEN = 512
EMBEDDING = 768

model = keras.Sequential()
model.add(Masking(input_shape=(MAX_SEQ_LEN,EMBEDDING)))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.load_weights(model_dir+"Ety=Gene_512_ECH=8_BCH=32_EMBD=768_3_layer_CRF.h5")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.summary()

X_te = np.load(os.path.join(embedding_dir, "BC2GM", "X_5test.npy"))
y_te = np.load(os.path.join(embedding_dir, "BC2GM", "y_5test.npy"))
pred_cat = model.predict(X_te)
y_pred = np.argmax(pred_cat, axis=-1)
y_true = np.argmax(y_te, -1)

get_measures(y_true, y_pred)
