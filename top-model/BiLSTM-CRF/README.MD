Run:

python3 train.py

train.py -> the driver code to train the BiLSTM model
model.py -> model architecture file
data_utils.py -> reads conll format huner datasets as lists of sentences/tags
data_gen.py -> converts the sentences/tags into train/test matrices
			to be used by keras model