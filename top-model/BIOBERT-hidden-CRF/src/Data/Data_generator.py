# -*- coding: utf-8 -*-

import numpy as np
import os
import keras

PATH = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/processed/"

class DataGenerator(keras.utils.Sequence):
  
    def __init__(self, dataset, batch_size, dim=(512,768),
                 n_classes=4, shuffle=False):
     
        self.path_train = PATH + dataset + "/Train/"
        self.path_label = PATH + dataset + "/Train_label/"   
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = [i for i in os.listdir(self.path_train) if i != ".ipynb_checkpoints"]
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 

        X_train = np.array([]).reshape(0, *(self.dim))
        y_train = np.array([]).reshape(0, self.dim[0], self.n_classes)
        
        for i, ID in enumerate(list_IDs_temp):

            y_ID = ID.split("_")[0] + "_lab_" + ID.split("_")[1].split(".")[0] + ".npy"

            x_bat = np.load(self.path_train + ID)
            y_bat = np.load(self.path_label + y_ID)
            X_train = np.concatenate((X_train, x_bat))
            y_train = np.concatenate((y_train, y_bat)) 
            
        return X_train, y_train
  