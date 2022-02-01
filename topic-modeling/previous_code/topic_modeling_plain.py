import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import glob
import collections
import re
import unicodedata
import six
import tensorflow as tf
from bs4 import BeautifulSoup
import copy
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import tqdm
import scipy
import umap
import seaborn as sns

from plain_preprocess import BasicTokenizer

B_tokenizer = BasicTokenizer()
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
punctuations = string.punctuation.replace('-','')
lemmatizer = WordNetLemmatizer()

def cleanText(data):
    data = re.sub(r'\([^()]*\)', '', data)
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')
    words = [s for s in B_tokenizer.tokenize(data) if re.match("^[A-Za-z0-9]+$", s)]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(s) for s in words]
    words = [s for s in words if not re.match("^[0-9]+$", s)]
    words = [s for s in words if not len(s) == 1]

    return words

def read_articles(directory):
    fp = os.path.join(directory, '*.txt')
    acsfiles = glob.glob(fp)
    acsdata = []
    for filepath in acsfiles:
        with open(filepath, 'r') as f:
            data = f.read()
            acsdata.append(data)
    return acsdata


def topic_model(data_words, k):
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    corpus_train, corpus_test, idx_train, idx_test = train_test_split(corpus, range(len(corpus)), \
                                                              test_size=0.2, random_state=0)
    data_words_train = list(pd.Series(data_words)[idx_train])
    data_words_test = list(pd.Series(data_words)[idx_test])
    lda_model = gensim.models.LdaModel(corpus=corpus_train,
                                       id2word=id2word,
                                       num_topics=k,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return corpus_test, idx_test, lda_model, coherence_lda


def main(targets):
    lemmatization = True
    acsdata = read_articles("txt-files/research")
    data_words = []
    for file in acsfiles:
        data_words.append(cleanText(file))
    k = 15
    corpus_test, idx_test, lda_model, coherence_lda = topic_model(data_words, k = k)
    print('Coherence Score: ', coherence_lda)
    for i in range(k):
        print('%d: '%i + ', '.join(re.findall(r'"(.*?)"', lda_model.print_topics()[i][1])))

    i = 0
    idx = idx_test[i]
    print(acsfiles[idx], '\n')
    filepath = acsfiles[idx]
    with open(filepath, 'r') as f:
        data = f.read()
    fid = filepath.replace('txt-files/research', '').replace('.txt', '')
    xml_path = 'acs' + fid*2 + '.xml'
    freq_list = id2word.doc2bow(cleanText(acsdata[idx]))
    distribution = lda_model.get_document_topics(freq_list)

    print('Topic Distribution for the Document:', '\n')
    for idx, perc in sorted(distribution, key = lambda x: x[1], reverse = True)[:5]:
        topic_terms = lda_model.get_topic_terms(idx)
        print(perc, '*', [(id2word[t[0]], 1 if id2word[t[0]] in data else 0,1 if id2word[t[0]] in path2entity[filepath] else 0) for t in topic_terms], '(Topic %d)'%idx, '\n')

    with open(xml_path, 'r') as f:
        urlText = f.read()
        soup = BeautifulSoup(urlText, features="lxml")
    keywords = [kwd.get_text() for kwd in soup.find('kwd-group')('kwd')]
    print('Keywords for the Document:', '\n', keywords)
