# author - Samantha Mahendran
# inspired by Lu, et al.-- https://github.com/Louis-udm/VGCN-BERT
import pandas as pd
import numpy as np
import scipy.sparse as sp
from math import log


class BuildGraph:
    def __init__(self, model, cleanData, valid_data_taux=0.2, tfidf_mode='only_tf', window_size=20, freq_min_for_word_choice=5,):
        self.data_model = model
        self.cleanData = cleanData

        all_train_size = len(self.cleanData.all_train_data)
        print('all train size', all_train_size)
        test_size = len(self.cleanData.test_data)
        valid_size = int(all_train_size * valid_data_taux)
        train_size = all_train_size - valid_size
        print('train, valid, test', train_size, valid_size, test_size)

        corpus_size = len(self.cleanData.labels)
        y_str = pd.Series(self.cleanData.labels)
        y_prob = pd.get_dummies(y_str).values
        self.class_labels = list(pd.get_dummies(y_str).columns)

        y = []
        for i in y_str.values:
            y.append(self.class_labels.index(i))
        y = np.asarray(y)
        shuffled_clean_docs = self.cleanData.clean_docs
        train_docs = shuffled_clean_docs[:train_size]
        valid_docs = shuffled_clean_docs[train_size:train_size + valid_size]
        train_valid_docs = shuffled_clean_docs[:train_size + valid_size]
        train_y = y[:train_size]
        valid_y = y[train_size:train_size + valid_size]
        test_y = y[train_size + valid_size:]
        train_y_prob = y_prob[:train_size]
        valid_y_prob = y_prob[train_size:train_size + valid_size]
        test_y_prob = y_prob[train_size + valid_size:]

        # build vocab using whole corpus(train+valid+test+genelization)
        word_set = set()
        for doc_words in shuffled_clean_docs:
            words = doc_words.split()
            for word in words:
                word_set.add(word)

        vocab = list(word_set)
        vocab_size = len(vocab)

        vocab_map = {}
        for i in range(vocab_size):
            vocab_map[vocab[i]] = i

        # build vocab_train_valid
        word_set_train_valid = set()
        for doc_words in train_valid_docs:
            words = doc_words.split()
            for word in words:
                word_set_train_valid.add(word)
        vocab_train_valid = list(word_set_train_valid)
        vocab_train_valid_size = len(vocab_train_valid)

        # %%
        # a map for word -> doc_list
        if tfidf_mode == 'all_tf_train_valid_idf':
            for_idf_docs = train_valid_docs
        else:
            for_idf_docs = shuffled_clean_docs
        word_doc_list = {}
        for i in range(len(for_idf_docs)):
            doc_words = for_idf_docs[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)

        '''
        Doc word heterogeneous graph
        and Vocabulary graph
        '''
        print('Calculate First isomerous adj and First isomorphic vocab adj, get word-word PMI values')

        adj_y = np.hstack((train_y, np.zeros(vocab_size), valid_y, test_y))
        adj_y_prob = np.vstack(
            (train_y_prob, np.zeros((vocab_size, len(self.class_labels)), dtype=np.float32), valid_y_prob, test_y_prob))

        windows = []
        for doc_words in train_valid_docs:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)

        print('Train_valid size:', len(train_valid_docs), 'Window number:', len(windows))

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        word_pair_count = {}
        for window in windows:
            appeared = set()
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = vocab_map[word_i]
                    word_j = window[j]
                    word_j_id = vocab_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in appeared:
                        continue
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    appeared.add(word_pair_str)
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in appeared:
                        continue
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    appeared.add(word_pair_str)

        row = []
        col = []
        weight = []
        tfidf_row = []
        tfidf_col = []
        tfidf_weight = []
        vocab_adj_row = []
        vocab_adj_col = []
        vocab_adj_weight = []

        num_window = len(windows)
        tmp_max_npmi = 0
        tmp_min_npmi = 0
        tmp_max_pmi = 0
        tmp_min_pmi = 0
        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            # 使用normalized pmi:
            npmi = log(1.0 * word_freq_i * word_freq_j / (num_window * num_window)) / log(1.0 * count / num_window) - 1
            if npmi > tmp_max_npmi: tmp_max_npmi = npmi
            if npmi < tmp_min_npmi: tmp_min_npmi = npmi
            if pmi > tmp_max_pmi: tmp_max_pmi = pmi
            if pmi < tmp_min_pmi: tmp_min_pmi = pmi
            if pmi > 0:
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(pmi)
            if npmi > 0:
                vocab_adj_row.append(i)
                vocab_adj_col.append(j)
                vocab_adj_weight.append(npmi)
        print('max_pmi:', tmp_max_pmi, 'min_pmi:', tmp_min_pmi)
        print('max_npmi:', tmp_max_npmi, 'min_npmi:', tmp_min_npmi)

        # %%
        print('Calculate doc-word tf-idf weight')

        n_docs = len(shuffled_clean_docs)
        doc_word_freq = {}
        for doc_id in range(n_docs):
            doc_words = shuffled_clean_docs[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = vocab_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(n_docs):
            doc_words = shuffled_clean_docs[i]
            words = doc_words.split()
            doc_word_set = set()
            tfidf_vec = []
            for word in words:
                if word in doc_word_set:
                    continue
                j = vocab_map[word]
                key = str(i) + ',' + str(j)
                tf = doc_word_freq[key]
                tfidf_row.append(i)
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                tfidf_col.append(j)
                col.append(train_size + j)
                # smooth
                idf = log((1.0 + n_docs) / (1.0 + word_doc_freq[vocab[j]])) + 1.0
                # weight.append(tf * idf)
                if tfidf_mode == 'only_tf':
                    tfidf_vec.append(tf)
                else:
                    tfidf_vec.append(tf * idf)
                doc_word_set.add(word)
            if len(tfidf_vec) > 0:
                weight.extend(tfidf_vec)
                tfidf_weight.extend(tfidf_vec)

        '''
        Assemble adjacency matrix and dump to files
        '''
        node_size = vocab_size + corpus_size

        adj_list = []
        adj_list.append(sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size), dtype=np.float32))
        for i, adj in enumerate(adj_list):
            adj_list[i] = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj_list[i].setdiag(1.0)

        vocab_adj = sp.csr_matrix((vocab_adj_weight, (vocab_adj_row, vocab_adj_col)), shape=(vocab_size, vocab_size),
                                  dtype=np.float32)
        vocab_adj.setdiag(1.0)

        # %%
        print('Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf...')
        tfidf_all = sp.csr_matrix((tfidf_weight, (tfidf_row, tfidf_col)), shape=(corpus_size, vocab_size),
                                  dtype=np.float32)
        tfidf_train = tfidf_all[:train_size]
        tfidf_valid = tfidf_all[train_size:train_size + valid_size]
        tfidf_test = tfidf_all[train_size + valid_size:]
        tfidf_X_list = [tfidf_train, tfidf_valid, tfidf_test]
        vocab_tfidf = tfidf_all.T.tolil()
        for i in range(vocab_size):
            norm = np.linalg.norm(vocab_tfidf.data[i])
            if norm > 0:
                vocab_tfidf.data[i] /= norm
        vocab_adj_tf = vocab_tfidf.dot(vocab_tfidf.T)

        # check
        print('Check adjacent matrix...')
        for k in range(len(adj_list)):
            count = 0
            for i in range(adj_list[k].shape[0]):
                if adj_list[k][i, i] <= 0:
                    count += 1
                    print('No.%d adj, abnomal diagonal found, No.%d' % (k, i))
            if count > 0:
                print('No.%d adj, totoal %d zero diagonal found.' % (k, count))

        self.vocab_map = vocab_map
        self.vocab = vocab
        self.adj_list = adj_list
        self.y = y
        self.y_prob = y_prob
        self.train_y = train_y
        self.train_y_prob = train_y_prob
        self.valid_y = valid_y
        self.valid_y_prob = valid_y_prob
        self.test_y = test_y
        self.test_y_prob = test_y_prob
        self.tfidf_X_list = tfidf_X_list
        self.vocab_adj = vocab_adj
        self.vocab_adj_tf = vocab_adj_tf
        self.shuffled_clean_docs = shuffled_clean_docs


