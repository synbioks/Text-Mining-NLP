import pickle 
from sklearn.model_selection import train_test_split
import gensim.corpora as corpora
import gensim 
from gensim.models import LdaSeqModel

class model():
    def __init__(self, data_words_path, model_name, k, path_to_mallet, model_path, a_sq=None, pub_year=None, test_size=0.2, random_seed=100, iterations=1000):
        self.data_words_path = data_words_path
        self.model_name = model_name
        self.k = k
        self.path_to_mallet = path_to_mallet
        self.model_path = model_path 
        self.test_size = test_size
        self.random_seed = random_seed
        self.iterations = iterations
        self.a_sq = a_sq 
        self.pub_year = pub_year 
        self._load_data_words()
        self._train_model()
        self._save_model()

    def _load_data_words(self):
        with open(self.data_words_path, "rb") as fp:
            self.data_words = pickle.load(fp)

    def _train_model(self):
        # Create Dictionary
        id2word = corpora.Dictionary(self.data_words)

        # Create Corpus
        texts = self.data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Train Test Split 
        corpus_train = train_test_split(
            corpus, range(len(corpus)), test_size=self.test_size, random_state=self.random_seed
        )[0]

        if self.model_name == 'lda':
            # Model 
            self.model = gensim.models.wrappers.LdaMallet(self.path_to_mallet, 
                                                    corpus=corpus_train, 
                                                    num_topics=self.k, 
                                                    id2word=id2word, 
                                                    random_seed=self.random_seed,
                                                    iterations=self.iterations)
        elif self.model_name == 'dtm':
            self.model = LdaSeqModel(
                            corpus=corpus_train, 
                            time_slice=year_slice, 
                            id2word=id2word, 
                            lda_model=lda_model, 
                            num_topics=self.k
                        )
        
    
    def _save_model(self):
        self.model.save(self.model_path)

def run_model(data_words_path, model_name, k, path_to_mallet, model_path):
    m = model(data_words_path, model_name, k, path_to_mallet, model_path)



        
        