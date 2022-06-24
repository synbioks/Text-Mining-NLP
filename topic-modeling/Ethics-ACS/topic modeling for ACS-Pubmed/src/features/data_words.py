from src.data.load_data import *
from src.data import clean
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_json(json_input_path):
    json_files = load_json_files(json_input_path)

    return json_files

def create_data_words(json_files, q):
    article_sq = []

    data_words = []
    for k in json_files.keys():
        article_sq.append(k)
        data_words.append(clean.NonNerCleanText(json_files[k]))

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

def save_data_words(json_input_path, q, output_path_data_words, output_path_a_sq):
    json_files = load_json(json_input_path)
    data_words, a_sq = create_data_words(json_files, q)
    pickle.dump(data_words, open(output_path_data_words, "wb"))
    pickle.dump(a_sq, open(output_path_a_sq, "wb"))

    

