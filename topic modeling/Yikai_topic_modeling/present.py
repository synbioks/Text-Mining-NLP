import pandas as pd
import numpy as np
import seaborn as sns
import re
import pickle as pickle
import json
from scipy.stats import entropy
import clean
from load_articles_entities import load_articles

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel

#sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from load_articles_entities import load_articles

def topics_df(model, bar=0.005, num_words=50):
    """
    Use this method to display topics in one DataFrame

    @param
    model: input model 
    bar: threshold of word frequency 
    num_words: number of words for each topic
    """
    topics = {}
    for idx, item in model.print_topics(num_words=num_words):
        idx += 1
        topic = []
        for num, j in zip(re.findall(r'\s(0.0[0-9]{2})\*', item),re.findall('\"([a-zA-Z]+)\"', item)):
            
            # set the threshold
            if float(num) >= bar:
                topic.append(j)
        topics[f'Topic{idx}'] = ', '.join(topic)

    # create the dataframe 
    df = pd.DataFrame(topics.values(), columns=['Terms per Topic'], index=topics.keys())
    return df 

def find_docs(model):
    """
    Return the topic distributions over each document
    """
    docs = []
    for i in model.load_document_topics():
        docs.append([j[1] for j in i])
    docs = np.array(docs)
    return docs 


def plot_cov(docs):
    """
    Plot the Topic Coverage
    """
    df = pd.DataFrame()
    count = 1
    for i in docs.T:
        df[f'Topic{count}'] = i
        count += 1
    ax = sns.barplot(
        x=df.columns, 
        y=df.sum().values,
        palette="deep"
    )
    return ax

def convertldaMalletToldaGen(mallet_model):
    """
    convert mallet_lda model to gensim_lda model 
    """
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) 
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

def save_corpus():
    """
    save corpus 
    """
    data_words_path = 'data/no_ner_data_words.txt'
    corpus_path = 'data/no_ner_corpus.pickle'
    # load data_words 
    with open(data_words_path, "rb") as fp:
        data_words = pickle.load(fp)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    pickle.dump(corpus, open(corpus_path, "wb"))

    return 

def contribution_df(model, corpus):
    # load articles 
    acsfiles = load_articles("data/txt-files-new/research/*.txt")[0]
    # load article titles 
    with open('data/acs_research_title.json', 'r') as fp:
        acs_title = json.load(fp)
    title_dict = {}
    for key in acs_title.keys():
        k = key[22:31]
        title_dict[k] = acs_title[key]

    # find topic distributions over documents 
    tm_results = model[corpus]
    corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]

    # create a dataframe
    df = pd.DataFrame()
    # Load Titles 
    df['ID'] = [a[28:37] for a in acsfiles]
    df['Title'] = df['ID'].map(title_dict)
    df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
    df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
    
    return df

def find_closest(df, node, nodes):
    #node = [year_df[year_df.article==name].doc.values[0]][0]
    min_value = float('inf')
    
    for ind, item in enumerate(nodes):
        if min_value > entropy(node, item) and entropy(node, item) != 0:
            min_value = entropy(node, item)
            min_item = ind
    #print(min_item)
    return df.iloc[min_item, :]

def parts_rich_df(model, k_top, id_df):

    # Load Title Dictionary 
    with open('data/acs_research_title.json', 'r') as fp:
        acs_title = json.load(fp)
    title_dict = {}
    for key in acs_title.keys():
        k = key[22:31]
        title_dict[k] = acs_title[key]

    # Parts Rich Article List 
    parts_rich = [
            'research/sb500366v.txt', # in the training data 
            'research/sb6b00337.txt', # in the training data 
            'research/sb9b00176.txt', 
            'non-research/sb5b00124.txt',
            'non-research/sb6b00031.txt',
            'non-research/sb8b00251.txt',
            'non-research/sb4001504.txt'
    ]
    parts_rich = [f'data/txt-files-new/{i}' for i in parts_rich]

    # Create one DataFrame 
    p = pd.DataFrame()

    # Load ID and Title 
    p['id'] = [re.findall(r'(sb[0-9a-z]+)', i)[0] for i in parts_rich]
    p['Title'] = p.id.map(title_dict)
    p_results = []
    for a in parts_rich:
        acsdata = load_articles(a)[1][0]
        bow = model.id2word.doc2bow(clean.NonNerCleanText(acsdata))
        p_results.append(model[bow])
    corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in p_results]
    p['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
    p['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]

    # Load Nodes Data 
    node_path = f'data/lda_k={k_top}_nodes.pickle'

    with open(node_path, "rb") as fp: 
        nodes = pickle.load(fp)

    close_nodes = []
    for n in p_results:
        n = find_closest(id_df, [i[1] for i in n], nodes)
        close_nodes.append(n.Title)
    
    p['Similar Article'] = close_nodes

    return p
    