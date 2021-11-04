import spacy
import json

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        obj = json.load(fin)
    return obj

def save_json(filename, obj):
    with open(filename, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout, sort_keys=True, indent=4)

def read_tsv(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            lines.append(line.strip().split('\t'))
    return lines

# lazy tokenzier load
tokenizer = None
def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = spacy.load('en_core_sci_sm')
    return tokenizer
