# author - Samantha Mahendran
# inspired by Lu, et al.-- https://github.com/Louis-udm/VGCN-BERT
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer


def del_http_user_tokenize(tweet):
    # delete [ \t\n\r\f\v]
    space_pattern = r'\s+'
    url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    tweet = re.sub(space_pattern, ' ', tweet)
    tweet = re.sub(url_regex, '', tweet)
    tweet = re.sub(mention_regex, '', tweet)
    return tweet


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class DataCleaning:

    def __init__(self, model, bert_model_scale='bert-base-uncased', del_stop_words=False, use_bert_tokenizer_at_clean=True, bert_lower_case = True):
        """
        :param model: model class object
        """
        self.model = model
        self.all_train_data = shuffle(self.model.meta_train_data)
        self.test_data = shuffle(self.model.meta_test_data)

        data = self.all_train_data  + self.test_data
        corpus = []
        labels = []
        for entry in data:
            corpus.append(entry.split('\t')[0])
            labels.append(entry.split('\t')[1])

        self.corpus = corpus
        self.labels = labels

        max_len_seq = 0
        max_len_seq_idx = -1
        min_len_seq = 1000
        min_len_seq_idx = -1
        sen_len_list = []
        for i, seq in enumerate(corpus):
            seq = seq.split()
            sen_len_list.append(len(seq))
            if len(seq) < min_len_seq:
                min_len_seq = len(seq)
                min_len_seq_idx = i
            if len(seq) > max_len_seq:
                max_len_seq = len(seq)
                max_len_seq_idx = i
        print('Statistics for original text: max_len%d,id%d, min_len%d,id%d, avg_len%.2f' \
              % (max_len_seq, max_len_seq_idx, min_len_seq, min_len_seq_idx, np.array(sen_len_list).mean()))

        if del_stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            stop_words = set(stop_words)
        # else:
            stop_words = {}
        # print('Stop_words:', stop_words)

        tmp_word_freq = {}  # to remove rare words
        new_doc_content_list = []

        # use bert_tokenizer for split the sentence
        if use_bert_tokenizer_at_clean:
            print('Use bert_tokenizer for seperate words to bert vocab')
            bert_tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=bert_lower_case)
        for doc_content in corpus:
            new_doc = doc_content
            # new_doc = clean_str(doc_content)
            if use_bert_tokenizer_at_clean:
                sub_words = bert_tokenizer.tokenize(new_doc)
                sub_doc = ' '.join(sub_words).strip()
                new_doc = sub_doc
            new_doc_content_list.append(new_doc)
            for word in new_doc.split():
                if word in tmp_word_freq:
                    tmp_word_freq[word] += 1
                else:
                    tmp_word_freq[word] = 1

        doc_content_list = new_doc_content_list

        # for normal dataset
        clean_docs = []
        count_void_doc = 0
        for i, doc_content in enumerate(doc_content_list):
            words = doc_content.split()
            doc_words = []
            for word in words:
                doc_words.append(word)
            doc_str = ' '.join(doc_words).strip()
            if doc_str == '':
                count_void_doc += 1
                print('No.', i, 'is a empty doc after treat, replaced by \'%s\'. original:%s' % (doc_str, doc_content))
            clean_docs.append(doc_str)

        print('Total', count_void_doc, ' docs are empty.')

        min_len = 10000
        min_len_id = -1
        max_len = 0
        max_len_id = -1
        aver_len = 0

        for i, line in enumerate(clean_docs):
            temp = line.strip().split()
            aver_len = aver_len + len(temp)
            if len(temp) < min_len:
                min_len = len(temp)
                min_len_id = i
            if len(temp) > max_len:
                max_len = len(temp)
                max_len_id = i

        aver_len = 1.0 * aver_len / len(clean_docs)
        self.clean_docs = clean_docs

        print('After tokenizer:')
        print('Min_len : ' + str(min_len) + ' id: ' + str(min_len_id))
        print('Max_len : ' + str(max_len) + ' id: ' + str(max_len_id))
        print('Average_len : ' + str(aver_len))