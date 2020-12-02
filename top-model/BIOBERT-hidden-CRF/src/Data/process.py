
"""
Convert CONLL files to two tsv files. 
train.tsv: Nounphrase removed
train_benchmark.tsv: Nounphrase kept

"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "Chemicals/" # datasets in NER/data/raw
ROOT_DIR = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/raw/"
DATASETS = [x for x in os.listdir(ROOT_DIR+DATA_DIR)]
if(".DS_Store" in DATASETS):
    DATASETS.remove(".DS_Store")
if(".ipynb_checkpoints" in DATASETS):
    DATASETS.remove(".ipynb_checkpoints")
    
splits = ["train", "dev", "test"]
    
def process(fname):
    
    file = open(fname).read()
    lines = file.split("\n")
    token, tag = [], []
    
    # flag: True if previous line is '-DOCSTART- X X O'
    flag = False
    for line in lines:

        # if previous line is '-DOCSTART- X X O', then the current line will be an empty line
        if flag: 
            flag = False
        elif line:
            if line == '-DOCSTART- X X O':
                flag = True                 
            else: 
                line_split = line.split()
                
                # Handle edge case: skip
                if line_split[0] == "null":
                    continue
#                     line_split[0] = "empty"

                token.append(line_split[0])
                tag.append(line_split[-1])
                
        # If it's an empty line and previous line is not '-DOCSTART- X X O', add "" as sentence splitting mark.
        else: 
            token.append("")
            tag.append("")
    return token, tag 
    
    
for dataset in DATASETS:
            
    path = ROOT_DIR+DATA_DIR+dataset+"/"
    
    for split in splits:            
        f = path + "{}.conll.{}".format(dataset, split)
        
        tokens, tags = process(f)
        data = pd.DataFrame()
        
        data['token'] = tokens
        data['tag'] = tags
        
        # Compare NER results with benchmark, keep Nounphrase
        data.to_csv(path+split+"_benchmark.tsv", index=False, header=False, sep='\t')
        
        data['tag'] = data['tag'].replace(['B-NP'], 'B')
        data['tag'] = data['tag'].replace(['I-NP'], 'I')
        data.to_csv(path+split+".tsv", index=False, header=False, sep='\t')
            