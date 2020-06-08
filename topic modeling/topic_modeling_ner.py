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

from ner_preprocess import BasicTokenizer

B_tokenizer = BasicTokenizer()
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()

def cleanText(filepath, path2entity, separate, lemmatization):
    entities = list(path2entity[filepath])
    entities.sort(key=len, reverse=True)
    with open(filepath, 'r') as f:
        data = f.read()
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')
    terms = []
    for t in entities:
        if ' ' in t:
            tnew = t.replace(' ', '_')
            data = data.replace(t, tnew)
            terms.append(tnew)
        else:
            terms.append(t)
    terms.sort(key=len, reverse=True)
    words = [s for s in B_tokenizer.tokenize(data, terms) if not (len(s) == 1 and _is_punctuation(s))]
    words = [w for w in words if not w in stop_words]
    words_new = []
    for w in words:
        if w in terms:
            if separate:
                words_new.extend(w.split('_'))
            else:
                words_new.append(w)
        else:
            if lemmatization:
                words_new.append(lemmatizer.lemmatize(w))
            else:
                words_new.append(w)
    words = words_new
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


entity_types = ['cellline', 'species', 'gene', 'chemical']
path2entity = {}
for p in acsfiles:
    q = p.replace('txt-files/research', '').replace('.txt', '')
    entities = set()
    filepaths = ['ACS_huner_ents2/' + t + q + '_entities.txt' for t in entity_types]
    for fp in filepaths:
        with open(fp, 'r') as f:
            for l in f.readlines():
                entity = ' '.join(l.split()[:-1]).strip()
                if '?' in entity or entity in stop_words or entity in punctuations or len(entity) == 1:
                    continue
                entities.add(entity)
    path2entity[p] = entities

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

def reweight_data_words(data_words_i, filepath, path2entity, t):
    entities = list(path2entity[filepath])
    new_words = []
    for w in data_words_i:
        if w not in entities:
            new_words.append(w)
        else:
            new_words.extend([w] * t)
    return new_words

def reweight_entity(id2word, freq_list, filepath, path2entity, t):
    entities = list(path2entity[filepath])
    ids = id2word.doc2idx(list(path2entity[filepath]))
    ids = [d for d in ids if d >= 0]
    for word_id in ids:
        for i in range(len(freq_list)):
            pair = freq_list[i]
            if pair[0] == word_id:
                freq_list[i] = (pair[0], pair[1] * t)
                break
    return freq_list

def main(targets):
    separate = True
    lemmatization = True
    acsdata = read_articles("txt-files/research")
    data_words = []
    for file in acsfiles:
        data_words.append(cleanText(file, path2entity, separate = separate, lemmatization = lemmatization))

    if 'no_weighting' in targets:
        k = 15
        corpus_test, idx_test, lda_model, coherence_lda = topic_model(data_words, k = k)
        print('Coherence Score: ', coherence_lda)
        for i in range(k):
            print('%d: '%i + ', '.join(re.findall(r'"(.*?)"', lda_model.print_topics()[i][1])))

        i = 0
        idx = idx_test[0]
        print(acsfiles[idx], '\n')
        filepath = acsfiles[idx]
        with open(filepath, 'r') as f:
            data = f.read()
        fid = filepath.replace('txt-files/research', '').replace('.txt', '')
        xml_path = 'acs' + fid*2 + '.xml'
        freq_list = id2word.doc2bow(cleanText(filepath, path2entity, separate, lemmatization))
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

    if '50TF' in targets:
        k = 15
        data_words_50 = [reweight_data_words(data_words[i], acsfiles[i], path2entity, 50) for i in range(len(acsfiles))]
        corpus_test, idx_test, lda_model, coherence_lda = topic_model(data_words_50, k = k)
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
        freq_list = id2word.doc2bow(cleanText(filepath, path2entity, separate, lemmatization))
        freq_list = reweight_entity(id2word, freq_list, filepath, path2entity, 50)
        distribution = lda_model.get_document_topics(freq_list)

        print('Topic Distribution for the Document:', '\n')
        for idx, perc in sorted(distribution, key = lambda x: x[1], reverse = True)[:5]:
            topic_terms = lda_model.get_topic_terms(idx)
            print(perc, '*', [(id2word[t[0]], 1 if id2word[t[0]] in data else 0, 1 if id2word[t[0]] in path2entity[filepath] else 0) for t in topic_terms], '(Topic %d)'%idx, '\n')

        with open(xml_path, 'r') as f:
            urlText = f.read()
            soup = BeautifulSoup(urlText, features="lxml")
        keywords = [kwd.get_text() for kwd in soup.find('kwd-group')('kwd')]
        print('Keywords for the Document:', '\n', keywords)


    if 'quantile' in targets:
        k = 15
        corpus = [' '.join(d) for d in data_words]
        tfidf = TfidfTransformer()
        vocabulary = list(set(' '.join(corpus).split()))
        pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
        ser = pd.Series(index = vocabulary, data = pipe['tfid'].idf_)
        stops = ser[ser<ser.quantile(0.001)].sort_values().index.tolist()
        data_words_cleaned = []
        for d in data_words:
            data_words_cleaned.append([w for w in d if w not in stops])
        corpus_test, idx_test, lda_model, coherence_lda = topic_model(data_words_cleaned, k = k)
        print('Coherence Score: ', coherence_lda)
        for i in range(k):
            print('%d: '%i + ', '.join(re.findall(r'"(.*?)"', lda_model.print_topics()[i][1])))

        lda_model = gensim.models.LdaModel.load("best_models/lda_model_k20_q0.001_lemma_separate_research")

        i = 0
        idx = idx_test[i]
        print(acsfiles[idx], '\n')
        filepath = acsfiles[idx]
        with open(filepath, 'r') as f:
            data = f.read()
        fid = filepath.replace('txt-files/research', '').replace('.txt', '')
        xml_path = 'acs' + fid*2 + '.xml'
        freq_list = id2word.doc2bow([w for w in cleanText(filepath, path2entity, separate, lemmatization) if w not in stops])
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
