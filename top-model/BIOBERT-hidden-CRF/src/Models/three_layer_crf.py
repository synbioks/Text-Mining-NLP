#!/usr/bin/env python
# coding: utf-8

"""
Model: BioBERT + three hidden layers + CRF 
comet_ml: Machine learning platform. https://www.comet.ml/site/
ENTITY: "Gene_512". The directory in NER/data/processed 
MAX_SEQ_LEN: 512
EMBEDDING: 768
BATCH_SIZE: if set to 2, processing 2*16 sentences each time. 
n_tags: 3

"""

# Import comet_ml in the top of your file(before all other Machine learning libs)
# from comet_ml import Experiment
# exp = Experiment(
#     # Place your own kep here
#     api_key="wGZl3crmGlYoTwvrvA6Tx46UG",
#     project_name='top-model Cluster')
# COMET_DISABLE_AUTO_LOGGING=1


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import sys

bert_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/bert" 
crf_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/keras-contrib/keras_contrib/layers" 
sys.path.append("/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/src")
sys.path.append(bert_dir)
sys.path.append(crf_dir)

from collections import defaultdict
from crf import CRF
from data import data_generator

print("bert->", bert_dir)
print("crf ->", crf_dir)

### Hyperparams
ENTITY = "Gene_512"
MAX_SEQ_LEN = 512
EMBEDDING = 768 
EPOCHS = 15
BATCH_SIZE = 2
n_tags = 3
LEARNING = 0.01

# Specify the parameters in Comel.ai 
# params = {
#   "batch_size":BATCH_SIZE*16,
#   "MAX_SEQ_LEN":MAX_SEQ_LEN,
#   "EMBEDDING": EMBEDDING,
#   "EPOCHS": EPOCHS
# }
# exp.log_parameters(params)

training_generator = data_generator.DataGenerator(ENTITY, BATCH_SIZE)
model = keras.Sequential()
model.add(Masking(input_shape=(MAX_SEQ_LEN,EMBEDDING))) # Masking layer 
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
# CRF Layer
crf = CRF(4)
model.add(crf)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
# model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    callbacks=[early_stop],
                    epochs=EPOCHS,
                    workers=6)

model_name = "Ety={0}_ECH={1}_BCH={2}_LR={3}_EMBD={4}_3_layer_CRF.h5".format(ENTITY, EPOCHS, BATCH_SIZE*16, LEARNING, EMBEDDING)
filepath="/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/models/"
model.save_weights(filepath+model_name)
