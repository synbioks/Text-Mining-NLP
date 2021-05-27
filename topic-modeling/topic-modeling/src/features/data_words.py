from src.data.load_data import *
from src.data import clean
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_json_ner(json_input_path, ner_input_path, contain_abstract):
    json_files = load_json_files(json_input_path, contain_abstract)
    path2entity = load_ner_entities(ner_input_path, json_files.keys())

    return json_files, path2entity

def create_data_words(json_files, path2entity, q):
    article_sq = []
    for k in json_files:
        if k not in path2entity:
            path2entity[k] = set()

    data_words = []
    for k in json_files.keys():
        article_sq.append(k)
        data_words.append(clean.NERCleanText(k, json_files[k], path2entity))
        #data_words.append(clean.NonNerCleanText(json_files[k]))

    if q > 0:
        corpus = [' '.join(d) for d in data_words]
        tfidf = TfidfTransformer()
        vocabulary = list(set(' '.join(corpus).split()))
        pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
        ser = pd.Series(index = vocabulary, data = pipe['tfid'].idf_)

        # create stop words list
        stops = ser[ser<ser.quantile(q)].sort_values().index.tolist()

        #update data_words 
        data_words_cleaned = []
        for d in data_words:
            data_words_cleaned.append([w for w in d if w not in stops])
        data_words = data_words_cleaned

    return data_words, article_sq

def save_data_words(json_input_path, ner_input_path, contain_abstract, q, output_path_data_words, output_path_a_sq):
    json_files, path2entity = load_json_ner(json_input_path, ner_input_path, contain_abstract)
    data_words, a_sq = create_data_words(json_files, path2entity, q)
    pickle.dump(data_words, open(output_path_data_words, "wb"))
    pickle.dump(a_sq, open(output_path_a_sq, "wb"))

    

