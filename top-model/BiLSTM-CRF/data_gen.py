import numpy as np
import os
import keras
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, sents, tags, word2idx, tag2idx, batch_size=32, dim=100,
                 n_classes=4, shuffle=False, fit=True ):
        """Initialization"""
        self.list_IDs = list_IDs
        self.sents = sents
        self.tags = tags
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.fit = fit
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.n_tags = n_classes
        self.MAX_LEN = dim
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization

        batch_sents = [self.sents[x] for x in list_IDs_temp]
        batch_tags = [self.tags[x] for x in list_IDs_temp]

        # Generate data
        X, y = self.convert_to_trainable(batch_sents, batch_tags)
        return X, y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # print("X.shape, y.shape = ", X.shape, y.shape)

        if(not self.fit):
            return X

        return X, y

    # def gaurav_pad_sequences(self, maxlen, sequences, padding="post", value=0):
    #     # print("maxlen=", maxlen)
    #     padded_sequences = []
    #     for seq in sequences:
    #         if(len(seq) < maxlen):
    #             padded_seq = seq + ([0] * (maxlen-len(seq)))
    #         elif(len(seq) > maxlen):
    #             padded_seq = seq[:maxlen]
    #         else:
    #             padded_seq = seq
    #         padded_sequences.append(padded_seq)
    #     return np.array(padded_sequences)

    def convert_to_trainable(self, args_sents, args_tags):
        # Convert each sentence from list of Token to list of word_index
        idx_sents = []
        for sent in args_sents:
            idx_sent = []
            for w in sent:
                if w not in self.word2idx:
                    w = "UNK"
                idx_sent.append(self.word2idx[w])
            idx_sents.append(idx_sent)
        X = pad_sequences(maxlen=self.MAX_LEN, sequences=idx_sents, padding="post", value=self.word2idx["PAD"])

        # Padding each sentence to have the same length
        y = pad_sequences(maxlen=self.MAX_LEN, sequences=args_tags, padding="post", value=self.tag2idx["PAD"]) 
        y = to_categorical(y, num_classes=self.n_tags)
        return X, y

 
