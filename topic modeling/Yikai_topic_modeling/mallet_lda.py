"""
Mallet LDA (https://radimrehurek.com/gensim/models/wrappers/ldamallet.html) 
is used in this file as the primary model to find the latent topics 
among ACS articles. 
In order to run this script, you need to download the mallet package from 
website. (I used mallet-2.0.8)

Author: Yikai Hao 
"""

import json # to load stored json files
import pickle as pickle # to load stored pickle files 
import re
import pandas as pd
import numpy as np

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import load_articles_entities as load_files
###############################
# Hyperparameters
###############################

quantile = 0.005 # top quantile% of general words will be added in to stop words 
k = 15 # number of topics 
seed = 100
co = 'c_v' # the way to calculate coherence score 
iterations = 1000 # number of iterations 

model_name = f'lda_k={k}_q={quantile}'
txt_path = "data/txt-files-new/research/*.txt"
path_to_mallet = "mallet-2.0.8/bin/mallet"
#ner_path = '../ACS_result/ACS_huner_ents1/' 
data_words_path = 'data/no_ner_data_words.txt'
corpus_path = f'data/no_ner_q={quantile}_corpus.pickle'
model_path = f'models/{model_name}.pickle'
co_path = f'models/{model_name}_coherence.pickle'

###############################
# Load files / Preprocess 
###############################
#acsfiles, acsdata = load_files.load_articles(txt_path)

# open data words pickle file to load data words if it is saved 
with open(data_words_path, "rb") as fp:
    data_words = pickle.load(fp)

###############################
# Stop Words 
###############################
# put the top general words into stop words based on tfidf
if quantile > 0:
    corpus = [' '.join(d) for d in data_words]
    tfidf = TfidfTransformer()
    vocabulary = list(set(' '.join(corpus).split()))
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
    ser = pd.Series(index = vocabulary, data = pipe['tfid'].idf_)

    # create stop words list 
    stops = ser[ser<ser.quantile(quantile)].sort_values().index.tolist()

    #update data_words 
    data_words_cleaned = []
    for d in data_words:
        data_words_cleaned.append([w for w in d if w not in stops])
    data_words = data_words_cleaned

###############################
# Model 
###############################
# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Train Test Split 
corpus_train = train_test_split(
    corpus, range(len(corpus)), test_size=0.2, random_state=0
)[0]

# Model 
model = gensim.models.wrappers.LdaMallet(path_to_mallet, 
                                         corpus=corpus_train, 
                                         num_topics=k, 
                                         id2word=id2word, 
                                         random_seed=seed,
                                         iterations=iterations)

###############################
# Save Models and Coherence Score 
###############################
# save corpus
pickle.dump(corpus, open(corpus_path, "wb"))

# save model 
pickle.dump(model, open(model_path, "wb"))
coherencemodel = CoherenceModel(
    model=model, texts=texts, dictionary=id2word, coherence=co
)
pickle.dump(coherencemodel, open(co_path, "wb"))



                    