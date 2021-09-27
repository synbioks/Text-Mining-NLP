#!/usr/bin/env python
# coding: utf-8

from os.path import isdir, isfile, join, abspath
import copy
import csv, sys
import spacy
from tqdm import tqdm
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='main',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

### Basic settings ###
parser.add_argument('--mode', choices=['training', 'development', 'test'], required=True)
args = parser.parse_args()
print(args)
mode = args.mode

dataset_dir = abspath("../datasets/drugprot-gs-training-development-test/{}/".format(mode))

abs_file = "drugprot_{}_abstracs.tsv".format(mode)
ent_file = "drugprot_{}_entities.tsv".format(mode)
rel_file = "drugprot_{}_relations.tsv".format(mode)


def label_mapper(label):
    if "GENE" in label:
        return "GENE"
    if "CHEMICAL" in label:
        return "CHEMICAL"
    return label

csv.field_size_limit(sys.maxsize)
entities = {}
with open(dataset_dir+ "/" + ent_file, "r", encoding="utf8") as fin:
    for row in csv.reader(fin, delimiter="\t"):
        #print(row)
        
        doc_id = int(row[0])
        ent_id = row[1]
        
        if doc_id not in entities:
            entities[doc_id] = []
            
        entities[doc_id].append({
            "label": label_mapper(row[2]),
            "start": int(row[3]),
            "end": int(row[4]),
            "id": ent_id,
            "txt": row[5]
        })
        #print(entities)

relns = {}
with open(dataset_dir+ "/" + rel_file, "r", encoding="utf8") as fin:
    print(mode, mode == 'test')
    if mode != "test":
        for row in csv.reader(fin, delimiter="\t"):
            doc_id = int(row[0])
            reln = row[1]
            arg1 = row[2][5:]
            arg2 = row[3][5:]
            if doc_id not in relns:
                relns[doc_id] = []
            
            relns[doc_id].append({
                "label": reln,
                "id": doc_id,
                "ent1": arg1,
                "ent2": arg2
            })

docs = {}
with open(dataset_dir+ "/" + abs_file, "r", encoding="utf8") as fin:
    for row in csv.reader(fin, delimiter="\t"):
        assert(len(row) == 3)
        doc_id = int(row[0])
        txt = row[1] + "\t" + row[2]
        
        if doc_id not in relns:
            docs[doc_id] = {}
        
        docs[doc_id] = {
            "id": doc_id,
            "txt": txt
        }


sents_len = [len(docs[doc]["txt"].split()) for doc in docs]
sents = [docs[doc]["txt"] for doc in docs]
sorted_sents = sorted(sents, key=lambda s: -len(s.split()))
tokenizer  = spacy.load("en_core_sci_sm")

sents = []
allsents_input = []

outfile = dataset_dir + "/" + "re_input_all.tsv"

# with open(dataset_dir + "/" + "re_input_all.tsv", "w", encoding="utf8") as ftest:
    # ftest.write("DocId\tSpanId\tEnt1\tEnt2\tSentence\tlbl\n")

for doc_id in tqdm(list(docs.keys())):
    # if doc_id not in relns:
    #     #print("no relation found for this doc")
    #     continue
    reln_list = relns.get(doc_id, [])
    curr_sent = docs[doc_id]["txt"]
    sent_spans = tokenizer(curr_sent).sents
    ent_type_to_list = {"GENE":[], "CHEMICAL":[]}
    for ent in entities.get(doc_id, []):
        ent_type_to_list[ent['label']].append(ent)

    for span_id, a in enumerate(sent_spans):
        #print(doc_id, str(a))
        sent_a = str(a)
        ent_pairs = [(copy.deepcopy(ent1), copy.deepcopy(ent2)) 
                        for ent1 in ent_type_to_list["GENE"] 
                        for ent2 in ent_type_to_list["CHEMICAL"]]
        for (ent1, ent2) in ent_pairs:
            lbl = "NA"
            for rel in reln_list:
                if ((ent1['id'] == rel["ent1"] and ent2['id'] == rel["ent2"]) or
                    (ent1['id'] == rel["ent2"] and ent2['id'] == rel["ent1"])):
                    lbl = rel["label"]

            if (ent1['start'] >= a.start_char and ent1['start'] < a.end_char and 
                ent2['start'] >= a.start_char and ent2['start'] < a.end_char):

                if ent1["start"] > ent2["start"]:
                    ent1, ent2 = ent2, ent1
                if ent1["start"] == ent2["start"] and lbl != "NA":
                    print("warning: same start positiion", doc_id)
                    print(doc_id, "  ------", ent1["txt"], "------", ent2["txt"], "-------", a)
                new_sent = [
                    sent_a[:ent1["start"]-a.start_char],
                    "@", ent1["label"], "$",
                    sent_a[ent1["end"]-a.start_char:ent2["start"]-a.start_char],
                    "@", ent2["label"], "$",
                    sent_a[ent2["end"]-a.start_char:]
                ]
                new_sent = "".join(new_sent)
                curr_out_sent = str(doc_id)  + "\t" + \
                                str(span_id) + "\t" + "Arg1:{}".format(ent1["id"]) + "\t" + "Arg2:{}".format(ent2["id"]) + "\t" + \
                                " ".join(new_sent.split()) + "\t" + lbl + "\n"
                curr_out_sent = [str(doc_id) , str(span_id), "Arg1:{}".format(ent1["id"]), "Arg2:{}".format(ent2["id"]), " ".join(new_sent.split()), lbl ]
                allsents_input.append(curr_out_sent)
                # ftest.write(curr_out_sent)
                assert(ent1["label"] != ent2["label"])
            elif lbl != "NA" and (ent1['start'] >= a.start_char and ent1['start'] < a.end_char and 
                not (ent2['start'] >= a.start_char and ent2['start'] < a.end_char)):
                print("warning : docid/lbl {}/{} entities not in the same sentence".format(doc_id, lbl))

df = pd.DataFrame(allsents_input, columns=["DocId", "SpanId", "Ent1", "Ent2", "Sentence", "lbl"])
df.to_csv(outfile, index=False, sep="\t")
