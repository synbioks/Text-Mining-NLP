"""
This file is used to extract the ACS articles’ ids and contents from the txt files 
and load the ner results.
"""

import glob 
from nltk.corpus import stopwords
import string


# set stop words 
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
punctuations = string.punctuation.replace('-','')

def load_articles(path):
    acsfiles = glob.glob(path)
    acsdata = []
    for filepath in acsfiles:
        with open(filepath, 'r') as f:
            data = f.read()
            acsdata.append(data)
    return acsfiles, acsdata

def load_entities(acsfiles, ent_path, old_data=False):
    entity_types = ['cellline', 'species', 'gene', 'chemical']
    path2entity = {}
    for p in acsfiles:
        q = p.replace('txt-files-new/research', '').replace('.txt', '')
        entities = set()
        filepaths = [ent_path + t + q + '_entities.txt' for t in entity_types]
        for fp in filepaths:
            try:
                with open(fp, 'r') as f:
                    for l in f.readlines():
                        if old_data:
                            entity = ' '.join(l.split()[:-1]).strip()
                        else:
                            entity = ' '.join(l.split()[:-1]).strip()
                        if '?' in entity or entity in stop_words or entity in punctuations or len(entity) == 1:
                            continue
                        entities.add(entity)
            except:
                continue
        path2entity[p] = entities
    return path2entity