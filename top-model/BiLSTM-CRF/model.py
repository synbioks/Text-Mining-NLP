from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

class BiLSTMModel:
	def __init__(self, MAX_SEQ_LEN, EMBEDDING, LSTM_HIDDEN_UNITS, LSTM_DENSE_DIM, n_words, n_tags):
		self.MAX_SEQ_LEN = MAX_SEQ_LEN
		self.EMBEDDING = EMBEDDING
		self.LSTM_HIDDEN_UNITS = LSTM_HIDDEN_UNITS
		self.LSTM_DENSE_DIM = LSTM_DENSE_DIM
		self.n_words = n_words
		self.n_tags = n_tags

	def define_model(self):
		input_layer = Input(shape=(self.MAX_SEQ_LEN,))
		model = Embedding(input_dim=self.n_words, output_dim=self.EMBEDDING, # actual n_words + 2 (PAD & UNK)
		                  input_length=self.MAX_SEQ_LEN)(input_layer)  # default: 300-dim embedding
		model = Bidirectional(LSTM(units=self.LSTM_HIDDEN_UNITS, return_sequences=True,
		                           recurrent_dropout=0.1))(model)  # variational biLSTM
		model = TimeDistributed(Dense(self.LSTM_DENSE_DIM, activation="relu"))(model)  # a dense layer as suggested by neuralNer
		crf = CRF(self.n_tags)  # CRF layer, actual n_tags+1(PAD)
		output_layer = crf(model)  # output

		model = Model(input_layer, output_layer)
		model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

		model.summary()

		return model