import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gensim
from itertools import combinations
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from topic_modeling_plain import cleanText, read_articles
%matplotlib inline

def c_v_topic_coherence(corex_model, corpus, texts, dictionary, topn):
    topics = corex_model.get_topics(n_words=topn)
    term_rankings = []
    for n,topic in enumerate(topics):
        topic_words,_ = zip(*topic)
        term_rankings.append(list(topic_words))
    cm = CoherenceModel(topics=term_rankings, corpus=corpus, texts = texts, dictionary=id2word, coherence='c_v')
    return cm.get_coherence()


acsdata = read_articles("txt-files/research")
data_words = []
for file in acsfiles:
    data_words.append(cleanText(file))

data = [' '.join(words) for words in data_words]
data_train, data_test, idx_train, idx_test = train_test_split(data, range(len(data)), \
                                                                  test_size=0.2, random_state=0)

id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]

vectorizer = CountVectorizer(stop_words='english', binary=True)
doc_word = vectorizer.fit_transform(data_train)
doc_word = ss.csr_matrix(doc_word)

words = list(np.asarray(vectorizer.get_feature_names()))

topic_model = ct.Corex(n_hidden= 25, words=words, max_iter=200, verbose=False)#, seed=1)
topic_model.fit(doc_word, words=words);
coherence = c_v_topic_coherence(topic_model, corpus = corpus, texts = texts, dictionary = id2word, topn = 20)
print('Coherence Score: ', coherence)

topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    print('{}: '.format(n) + ', '.join(topic_words))
