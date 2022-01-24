import csv
import os
from tqdm import tqdm
from argparse import ArgumentParser
from os.path import isdir, join

import spacy

def entity_pairs(buckets, labels):
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            bucket1 = buckets.get(labels[i], [])
            bucket2 = buckets.get(labels[j], [])
            for ent1 in bucket1:
                for ent2 in bucket2:
                    yield ent1, ent2

def label_mapper(label):
    if "GENE" in label:
        return "GENE"
    if "CHEMICAL" in label:
        return "CHEMICAL"
    return label

def preprocess_article(tokenizer, txt_path, ann_path, output_path):

    # read in the text file
    with open(txt_path, "r", encoding="utf8") as fin:
        txt_raw = fin.readlines()

    txt_raw = [x.strip() for x in txt_raw]
    txt_raw = " ".join(txt_raw)

    # use spacy to separate the sentences
    sent_spans = tokenizer(txt_raw).sents

    # readin the ann file
    entities = []
    with open(ann_path, "r", encoding="utf8") as fin:
        for row in csv.reader(fin, delimiter="\t"):
            if row[0][0] == "R":
                continue
            entity = row[1].split(" ")
            #print(entity)
            entities.append({
                "label": label_mapper(entity[0].upper()),
                "start": int(entity[1]),
                "end": int(entity[2]),
                "id": row[0],
                "text": row[2]
            })

    entities = sorted(entities, key=lambda ent: ent["start"])

    # there shouldn't overlapping entities
    for i in range(1, len(entities)):
        if entities[i]["start"] < entities[i-1]["end"]:
            print("[WARNING] entity overlap in {}:".format(ann_path))
            print("\t1st entity: {}".format(entities[i-1]).encode("utf8"))
            print("\t2nd entity: {}".format(entities[i]).encode("utf8"))

    # the next section anonymizes entities
    ent_ptr = 0
    with open(output_path, "w", encoding="utf8") as ftest:
        ftest.write("id1\tid2\tsentence\n")
        for span in sent_spans:

            # find what entities are in the current sentence
            sent = span.text
            start = span.start_char
            end = span.end_char
            buckets = {} # to hold all the entities in this sentence, grouped by label types
            while ent_ptr < len(entities):
                ent = entities[ent_ptr]
                if ent["start"] >= start and ent["end"] <= end:
                    if ent["label"] not in buckets:
                        buckets[ent["label"]] = []
                    buckets[ent["label"]].append(ent)
                    ent_ptr += 1
                else:
                    break

            # substitute entities with label
            # we are only interested in chamical and gene at the moment
            for ent1, ent2 in entity_pairs(buckets, ["CHEMICAL", "GENE"]):
                if ent1["start"] > ent2["start"]:
                    ent1, ent2 = ent2, ent1
                new_sent = [
                    sent[:ent1["start"]-start],
                    "@", ent1["label"], "$",
                    sent[ent1["end"]-start:ent2["start"]-start],
                    "@", ent2["label"], "$",
                    sent[ent2["end"]-start:],
                    "\n"
                ]
                new_sent = "".join(new_sent)
                ftest.write("".join([ent1["id"], "\t", ent2["id"], "\t", new_sent]))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--dataset-dir')
    args = parser.parse_args()

    # handle the paths
    article_paths = []
    for pub_num in os.listdir(args.dataset_dir):
        article_dir = join(args.dataset_dir, pub_num)
        assert(isdir(article_dir)), f'folder {article_dir} is not a valid article directory'
        article_paths.append({
            "txt_path": join(article_dir, f"{pub_num}.txt"),
            "ann_path": join(article_dir, f"{pub_num}.ann"),
            "output_path": join(article_dir, "re_input.tsv")
        })

    # process the data
    tokenizer = spacy.load("en_core_sci_sm")
    for item in tqdm(article_paths):
        preprocess_article(tokenizer, **item)
