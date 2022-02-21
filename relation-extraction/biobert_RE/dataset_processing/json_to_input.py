import argparse

from utils import utils

# this can be faster, but ents is usually very short
# search ent_id in the ents
# return the ent object
def get_ent_by_id(ent_id, ents):
    for ent in ents:
        if ent['id'] == ent_id:
            return ent
    assert False, 'failed to find matching entity'

# this can be faster, but rels is usually very short
# given known relations, return whether ent1 and ent2 are related
def is_related(ent1, ent2, rels):
    for rel in rels:
        cond1 = ent1['id'] == rel['ent_id1'] and ent2['id'] == rel['ent_id2']
        cond2 = ent1['id'] == rel['ent_id2'] and ent2['id'] == rel['ent_id1']
        if cond1 or cond2:
            return True
    return False

def is_chem(ent):
    return ent['type'] == 'CHEMICAL'

def is_gene(ent):
    return ent['type'] in ['GENE', 'GENE-Y', 'GENE-N']

# check if the entities are a pair of chemical and gene
# if there is an overlap between these two entities
# it is not a chem gene pair
def is_chem_gene_pair(ent1, ent2):
    # check overlap
    if ent1['start'] >= ent2['end'] or ent2['start'] >= ent1['end']:
        cond1 = is_chem(ent1) and is_gene(ent2)
        cond2 = is_chem(ent2) and is_gene(ent1)
        return cond1 or cond2
    else:
        return False

# mask the entities in the sentence
# return the masked sentence
def create_input_sent(sent, ent1, ent2):
    offset = sent['start']
    # make sure ent1 is the left entities
    if ent2['start'] < ent1['start']:
        ent1, ent2 = ent2, ent1
    res = [
        sent['text'][:ent1['start']-offset],
        '@GENE$' if is_gene(ent1) else '@CHEMICAL$',
        sent['text'][ent1['end']-offset:ent2['start']-offset],
        '@GENE$' if is_gene(ent2) else '@CHEMICAL$',
        sent['text'][ent2['end']-offset:]
    ]
    return ''.join(res)

def get_samples(json_data, target_id=None):

    # go through the relations in json and generate input sentences
    # we should only be looking at chemical/gene relations
    input_sent_pos = []
    input_sent_neg = []
    for article_id, article_data in json_data.items():

        # skip everything except target_id
        if target_id is not None and article_id != target_id:
            continue

        for sent_idx, sent in enumerate(article_data['abstract']):
            ents = sent['entities']
            rels = sent['relations']

            # this part generates positive samples
            for rel_idx, rel in enumerate(rels):
                ent_id1 = rel['ent_id1']
                ent_id2 = rel['ent_id2']
                ent1 = get_ent_by_id(ent_id1, ents)
                ent2 = get_ent_by_id(ent_id2, ents)
                # generate positive samples
                if is_chem_gene_pair(ent1, ent2):
                    input_id = f'{article_id}_{sent_idx}_{rel_idx}'
                    label = rel['rel_type']
                    input_sent = create_input_sent(sent, ent1, ent2)
                    original_sent = sent['text']
                    input_sent_pos.append((input_id, label, input_sent, original_sent))

            # generate negative samples
            # chem-gene pairs that are not in the relation are treated as negative samples
            for i in range(len(ents) - 1):
                for j in range(i, len(ents)):
                    ent1, ent2 = ents[i], ents[j]
                    if not is_related(ent1, ent2, rels) and is_chem_gene_pair(ent1, ent2):
                        input_id = f'{article_id}_{sent_idx}_{i}_{j}'
                        label = 'NOT'
                        input_sent = create_input_sent(sent, ent1, ent2)
                        original_sent = sent['text']
                        input_sent_neg.append((input_id, label, input_sent, original_sent))
    
    return input_sent_pos, input_sent_neg


def main(json_filename, out_filename, target_id):

    json_data = utils.read_json(json_filename)
    pos_samples, neg_samples = get_samples(json_data, target_id)
    all_samples = pos_samples + neg_samples
    utils.save_tsv(out_filename, all_samples)

if __name__ == '__main__':

    # generating training/testing data from json_file

    parser = argparse.ArgumentParser()
    parser.add_argument('json_filename')
    parser.add_argument('out_filename')
    parser.add_argument('-i', '--target_id') # only process one article with this id (useful for debugging)
    args = parser.parse_args()
    main(args.json_filename, args.out_filename, args.target_id)