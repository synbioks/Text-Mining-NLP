import json
import os

from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np 


def load_json_files(input_path, contain_abstract=True):
    """
    Load json files from data/json-files 
    Only read research articles 

    @param input_path: path to json-files 
    @param contain_abstract: contain abstract part or not, default is True

    @return json_files: e.x. {'sb9b00393': '...'}
    """
    json_files = {}
    walk_list = os.walk(input_path)
    for i in walk_list:
        root = i[0]
        for j in i[2]:
            p = root + '/' + j
            j_file = json.load(open(p))

            # only keep research articles 
            if j_file['is_research']:
                text = ''

                # contain abstract or not 
                if contain_abstract:
                    text += ' '.join(j_file['abstract'])

                # append text 
                for b in j_file['body']:
                    if 'text' in b:
                        text += ' '.join(b['text'])
                json_files[j.replace('.json', '')] = text
    return json_files

def load_ner_entities(input_path, article_ids):
    """
    Load NER entities of each article from NER results 

    @param input_path
    @param article_ids: e.x. ['sb9b00393', ...]

    @return path2entity: {'sb9b00393', {set of NER terms}}
    """

    # define stop_words and punctuations
    stop_words = set(stopwords.words('english'))

    # not include '-'
    punctuations = string.punctuation.replace('-','')

    path2entity = {}
    for i in article_ids:
        e = set()
        
        # get the .ann file path 
        p = input_path + '/' + i + '.ann'

        try: 
            # get the ner terms from .ann file 
            df = pd.read_csv(p, sep='\t', header=None)

            # remove nan 
            ner_array = df[df.iloc[:, -1].notnull()].iloc[:, -1].values
            for n in ner_array:

                # remove wired NER terms 
                if '?' in n or n in stop_words or n in punctuations or len(n) == 1:
                    continue
                else: 
                    e.add(n)

        # some articles don't have NER result 
        except:
            #print(p)
            continue 
    
        path2entity[i] = e 
    
    return path2entity 

