import os
import json
import pandas as pd

files = ["train", "dev", "test"]

def convert(lines, f):
    tokens_ = []
    tags_ = []

    data = {"words": [], "ner": []}

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            data["words"].append(tokens_)
            data["ner"].append(tags_)
            tokens_ = []
            tags_ = []
        else:
            token, tag = line.split("\t")
            if len(tag) > 1:
                tag = tag.split("-")[0]
            tokens_.append(token.strip())
            tags_.append(tag.strip())
            
    if len(tokens_) > 0:
        data["words"].append(tokens_)
        data["ner"].append(tags_)

    return data

def writer(data, fp, add_str=""):

    for (tokens, tags) in zip(data["words"], data["ner"]):
        for (token, tag) in zip(tokens, tags):
            if tag == "B" or tag == "I":
                tag += add_str
            fp.write("{}\t{}\n".format(token, tag))
        fp.write("\n")
        
def convert_tsv_to_txt(data_dir):
    ## convert all tsv files to txt
    all_data = {}
    for f in files:
        with open(os.path.join(data_dir, f + ".tsv"), "r") as fp:
            lines = fp.readlines()
            all_data[f] = convert(lines, fp)
        fp = open(os.path.join(data_dir, f + ".txt"), "w")
        writer(all_data[f], fp)
        fp.close()
        
    return all_data
        
def get_train_test_df(all_data):
    # add the index to keep track of sentences
    train_tuples = []
    for i,(tokens,tags) in enumerate(zip(all_data["train"]["words"],all_data["train"]["ner"])):
        for token,tag in zip(tokens,tags):
            train_tuples.append([i,token,tag])

    test_tuples = []
    for i,(tokens,tags) in enumerate(zip(all_data["test"]["words"],all_data["test"]["ner"])):
        for token,tag in zip(tokens,tags):
            test_tuples.append([i,token,tag])

    train_df = pd.DataFrame(train_tuples, columns=['sentence_id', 'words', 'labels'])
    test_df = pd.DataFrame(test_tuples, columns=['sentence_id', 'words', 'labels'])
    
    return train_df, test_df