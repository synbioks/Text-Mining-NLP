from os.path import basename
import argparse

from utils import utils
from utils import json_to_brat

def article_id_overlap_check(id_list1, id_list2):

    id_set1 = set(id_list1)
    id_set2 = set(id_list2)

    # number of items in list1, number of items in list2, union, intersection
    return len(id_set1), len(id_set2), len(id_set1.union(id_set2)), len(id_set1.intersection(id_set2))


def compare_sentence(sent1, sent2):
    return all([
        sent1['start'] == sent2['start'],
        sent1['end'] == sent2['end'],
        sent1['text'] == sent2['text']
    ])


def compare_relation(rel1, rel2):
    return all([
        rel1['start'] == rel2['start'],
        rel1['end'] == rel2['end'],
        rel1['ent_id1'] == rel2['ent_id1'],
        rel1['ent_id2'] == rel2['ent_id2'],
        rel1['rel_type'] == rel2['rel_type']
    ])


def compare_entity(ent1, ent2):
    return all([
        ent1['start'] == ent2['start'],
        ent1['end'] == ent2['end'],
        ent1['text'] == ent2['text'],
        ent1['type'] == ent2['type'],
        ent1['id'] == ent2['id']
    ])


# find the difference between the first list of objects and the second list of objects
def find_list_diff(l1, l2, cmp_fn):
    in1not2 = []
    in2not1 = []
    # should be a better way if we can hash these objects
    for i1 in l1:
        for i2 in l2:
            if cmp_fn(i1, i2):
                break
        else:
            in1not2.append(i1)
    for i2 in l2:
        for i1 in l1:
            if cmp_fn(i1, i2):
                break
        else:
            in2not1.append(i2)
    return in1not2, in2not1


def compare_abstract(abs1, abs2):

    reason = []

    # used to store difference in the entire abstract
    ents_in1not2 = []
    ents_in2not1 = []
    rels_in1not2 = []
    rels_in2not1 = []

    # check if they have same number of sentences
    if len(abs1) != len(abs2):
        reason.append(f'different number of sentences')
        return reason

    # check if each sentence is the same
    for sent1, sent2 in zip(abs1, abs2):
        if not compare_sentence(sent1, sent2):
            reason.append(f'sentences')
            reason.append(f'{sent1}')
            reason.append(f'{sent2}')
            continue
        
        # check if each sentence have the different entities
        ent_in1not2, ent_in2not1 = find_list_diff(sent1['entities'], sent2['entities'], compare_entity)
        ents_in1not2.extend(ent_in1not2)
        ents_in2not1.extend(ent_in2not1)

        # check if each sentence have the different relation
        rel_in1not2, rel_in2not1 = find_list_diff(sent1['relations'], sent2['relations'], compare_relation)
        rels_in1not2.extend(rel_in1not2)
        rels_in2not1.extend(rel_in2not1)
        
    return ents_in1not2, ents_in2not1, rels_in1not2, rels_in2not1

def print_diff(diff):
    ents_in1not2, ents_in2not1, rels_in1not2, rels_in2not1 = diff
    # print the diff
    if ents_in1not2:
        print('entities in 1 not in 2')
        for item in ents_in1not2:
            print(f'{item}')
    if ents_in2not1:
        print('entities in 2 not in 1')
        for item in ents_in2not1:
            print(f'{item}')
    
    if rels_in1not2:
        print('relations in 1 not in 2')
        for item in rels_in1not2:
            print(f'{item}')
    if rels_in2not1:
        print('relations in 2 not in 1')
        for item in rels_in2not1:
            print(f'{item}')

def main(json_filename1, json_filename2, brat_diff_dir=None):

    dataset1 = utils.read_json(json_filename1)
    dataset2 = utils.read_json(json_filename2)

    # calculate article overlaps
    ds1_size, ds2_size, unique_size, common_size = article_id_overlap_check(dataset1.keys(), dataset2.keys())
    print(f'number of articles in {basename(json_filename1)}: {ds1_size}')
    print(f'number of articles in {basename(json_filename2)}: {ds2_size}')
    print(f'number of unique articles in both files: {unique_size}')
    print(f'number of common articles in both files: {common_size}')

    # if an article is mentioned in both dataset, compare their abstract, relations, and entities
    print('trying to find article ids in both dataset')
    print(f'comparing {basename(json_filename1)} to {basename(json_filename2)}')
    for article_id, data in dataset1.items():
        abstract = data['abstract']
        if article_id in dataset2:
            abstract_other = dataset2[article_id]['abstract']
            diff = compare_abstract(abstract, abstract_other)
            if any([len(x) > 0 for x in diff]):
                print(f'article id {article_id}')
                print_diff(diff)
                if brat_diff_dir:
                    txt, ann = json_to_brat.article_brat_repr(data, include_entities=True)
                    rels_in1not2, rels_in2not1 = diff[2], diff[3]
                    for i, rel in enumerate(rels_in1not2):
                        ann.append(json_to_brat.rel_brat_repr(rel, i, type_suffix='_1'))
                    for i, rel in enumerate(rels_in2not1):
                        ann.append(json_to_brat.rel_brat_repr(rel, i + len(rels_in1not2), type_suffix='_2'))
                    json_to_brat.write_brat(article_id, brat_diff_dir, txt, ann)

if __name__ == '__main__':

    # compare two json datasets, check if same articles are mentioned
    # if yes, check if the entities and relations are the same

    parser = argparse.ArgumentParser()
    parser.add_argument('json_filename1')
    parser.add_argument('json_filename2')
    parser.add_argument('-d', '--brat_diff_dir')
    args = parser.parse_args()
    main(args.json_filename1, args.json_filename2, args.brat_diff_dir)
