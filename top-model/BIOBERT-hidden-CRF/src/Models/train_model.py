#!/usr/bin/env python
# coding: utf-8

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
from keras.callbacks import EarlyStopping
import sys

bert_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/bert" 
crf_dir = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/keras-contrib/keras_contrib/layers" 
sys.path.append("/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/src")
sys.path.append(bert_dir)
sys.path.append(crf_dir)

from crf import CRF
from data import Data_generator
print("bert->", bert_dir)
print("crf ->", crf_dir)

### Hyperparams
ENTITY = "Gene_512"
MAX_SEQ_LEN = 512
EMBEDDING = 768 
EPOCHS = 8
BATCH_SIZE = 2
N_tags = 3
CHUNk_SIZE = 16

training_generator = data_generator.DataGenerator(ENTITY, BATCH_SIZE)

model = keras.Sequential()
model.add(Masking(input_shape=(MAX_SEQ_LEN,EMBEDDING)))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
                    use_multiprocessing=True,
                    callbacks=[early_stop],
                    epochs=EPOCHS,
                    workers=6)

model_name = "Ety={0}_ECH={1}_BCH={2}_EMBD={3}_3_layer_CRF.h5".format(ENTITY, EPOCHS, BATCH_SIZE*CHUNk_SIZE, EMBEDDING)
filepath="/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/models/"
model.save_weights(filepath+model_name)
