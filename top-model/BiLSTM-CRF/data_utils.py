import os
from collections import defaultdict


class Dataset:
	def __init__(self, data_dir, ):
		self.data_dir = data_dir
		self.set_tag2idx()

	def get_sentences_helper(self, fname):
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
	            tag.append(self.tag2idx[tg])
	        else:
	            sentences.append(sentence)
	            tags.append(tag)
	            sentence = []
	            tag = []
	    
	    return sentences, tags

	def get_sentences(self):
	    sents = defaultdict(list)
	    tags = defaultdict(list)

	    datasets = os.listdir(self.data_dir)
	    if(".DS_Store" in datasets):
	        datasets.remove(".DS_Store")

	    splits = ["train", "val", "test"]

	    map_split = lambda x : "devel" if x =="val" else x

	    for dataset in datasets:
	        for split in splits:
	            file_path = os.path.join(self.data_dir, dataset, map_split(split) + ".tsv")
	            sents_func, tags_func = self.get_sentences_helper(file_path)

	            sents[split] += sents_func
	            tags[split] += tags_func

	    return sents, tags

	def get_all_data(self):
		sents, tags = self.get_sentences()

		self.train_sents = sents["train"]
		self.train_tags = tags["train"]
		self.val_sents = sents["val"]
		self.val_tags = tags["val"]
		self.test_sents = sents["test"]
		self.test_tags = tags["test"]

		self.set_word2idx()

		return sents, tags

	def set_word2idx(self):
		words = defaultdict(int)

		for sent_list, tag_list in zip(self.train_sents, self.train_tags):
		    for word, tag in zip(sent_list, tag_list):
		        words[word] += 1


		n_words = len(words)
		words_list = words.keys()
		self.word2idx = {w: i + 2 for i, w in enumerate(words_list)}
		self.word2idx["UNK"] = 1 # Unknown words
		self.word2idx["PAD"] = 0 # Padding

		# Vocabulary Key:token_index -> Value:word
		self.idx2word = {i: w for w, i in self.word2idx.items()}

	def set_tag2idx(self):
		# The first entry is reserved for PAD
		self.tag2idx = {"B" : 1, "I" : 2, "O" : 3}
		self.tag2idx["PAD"] = 0
		self.idx2tag = {i: w for w, i in self.tag2idx.items()}
		self.n_tags = len(self.tag2idx)

	def get_word2idx(self):
		return self.word2idx
	def get_tag2idx(self):
		return self.tag2idx
	def get_nwords(self):
		return len(self.word2idx)
	def get_ntags(self):
		return self.n_tags