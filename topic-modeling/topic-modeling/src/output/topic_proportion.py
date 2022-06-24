import os
import json
import pickle

def output_proportion_to_data(doc_lda_path, data_json_path, article_sequence_path, topic_interpretations):
    # load doc_lda
    with open(doc_lda_path, "rb") as fp:
        doc_lda = pickle.load(fp)
    
    # load article_sq
    with open(article_sequence_path, 'rb') as fp:
        article_sq = pickle.load(fp)

    # turn doc_lda to a list of topic proportions without topic index
    doc_lda_no_index = [[i[1] for i in j] for j in doc_lda]

    doc_lda_w_interpretation = [list(zip(d,topic_interpretations)) for d in doc_lda_no_index]

    # pair the topic proportions and interpretatons with the article key
    topic_distribution_file = dict(zip(article_sq, doc_lda_w_interpretation))


    data_in_jsons = os.listdir(data_json_path)
    
    for file in data_in_jsons:
        
        json_path = os.path.join(data_json_path, file)
        # read original json in data
        with open(json_path, 'r') as f:
            d = json.load(f)
            # topic modeling is only done on research articles
            if d['is_research']:
                d['topic_modeling'] = topic_distribution_file[file.split('.')[0]]

        # write to update the json file
        with open(json_path, 'w') as f:
            json.dump(d, f, indent = 5)

    