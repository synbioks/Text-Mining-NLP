import argparse

from spacy import util

from utils import utils
from utils import cpr

def relation_key(rel):
    # create a string representation for rel
    # useful for merging relation
    return f'{rel["start"]} {rel["end"]} {rel["ent_id1"]} {rel["ent_id2"]}'

# in place, map all relations in the dataset to CPR-X
def map_to_cpr(dataset):
    for article_id, article in dataset.items():
        for sent in article['abstract']:
            for rel in sent['relations']:
                rel['rel_type'] = cpr.cpr_map[rel['rel_type']]

def merge_article(target, source):
    for trg_sent, src_sent in zip(target['abstract'], source['abstract']):
        merged_relations = {}
        for trg_rel in trg_sent['relations']:
            # trg_rel['rel_type'] = cpr.cpr_map[trg_rel['rel_type']] # convert to CPR-X
            trg_key = relation_key(trg_rel)

            # look for duplicate relations
            if trg_key in merged_relations:
                print('two relations found for one pair of entities')
                print(merged_relations[trg_key])
                print(trg_rel)
                continue
            merged_relations[trg_key] = trg_rel
        for src_rel in src_sent['relations']:
            # src_rel['rel_type'] = cpr.cpr_map[src_rel['rel_type']] # convert to CPR-X
            src_key = relation_key(src_rel)
            if src_key in merged_relations:
                # try to merge two relation together
                if merged_relations[src_key]['rel_type'] != src_rel['rel_type']:
                    print('failed to merge')
                    print(merged_relations[src_key])
                    print(src_rel)
            else:
                # this is a new relation unseen in target
                merged_relations[src_key] = src_rel

        trg_sent['relations'] = list(merged_relations.values())


def merge_datasets(dataset_filenames, out_filename):
    # the assumption here is that the abstract text and entity detected are identical
    # the only different things are relations
    # for chemprot and drugprot this is the case, see compare_datasets.py
    datasets = []
    for filename in dataset_filenames:
        datasets.append(utils.read_json(filename))
    # map all dataset to CPR-X
    for ds in datasets:
        map_to_cpr(ds)
    merged = datasets[0]
    for ds in datasets[1:]:
        for article_id, article in ds.items():

            # if this article is unique to this dataset, just add it
            # but make sure its relations are converted to CPR-X
            if article_id not in merged:
                merged[article_id] = article
            else:
                print(f'merging {article_id}')
                merge_article(merged[article_id], article)
    
    # stats
    total_relation = 0
    for article_id, article in merged.items():
        for sent in article['abstract']:
            total_relation += len(sent['relations'])
    print(f'number of relation of merged dataset: {total_relation}')

    utils.save_json(out_filename, merged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    merge_datasets(args.datasets, args.output)