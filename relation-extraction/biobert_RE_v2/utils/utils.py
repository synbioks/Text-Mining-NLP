import spacy
import json

def read_tsv(filename, encoding='ascii'):
    lines = []
    with open(filename, 'r', encoding=encoding) as fin:
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

def save_json(filename, obj):
    with open(filename, 'w') as fout:
        json.dump(obj, fout, sort_keys=True, indent=4)