Run:
<br>
python3 train.py
<br>
<br>
train.py -> the driver code to train the BiLSTM model
<br>
model.py -> model architecture file
<br>
data_utils.py -> reads conll format huner datasets as lists of sentences/tags
<br>
data_gen.py -> converts the sentences/tags into train/test matrices
			to be used by keras model

Data can be found at:
<br>
/sbksvol/gaurav/BiLSTM-CRF/data
<br>
To run the code, place all the 4 files in a folder say, "test" and the data folder also within test. Then run train.py
