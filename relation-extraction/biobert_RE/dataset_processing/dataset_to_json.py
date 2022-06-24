from tqdm import tqdm
import argparse

from utils import utils

# {article_id: {ent_id: {entity information}}}
def read_ent_file(filename):
    results = {}
    lines = utils.read_tsv(filename)
    for line in lines:

        assert len(line) == 6, 'each line of entity files should have 6 columns'
        article_id, ent_id, ent_type, start, end, text = line
        if args.debug is not None and article_id != args.debug:
            continue

        if article_id not in results:
            results[article_id] = {}
        ents = results[article_id]
        assert ent_id not in ents, 'entity id should not repeat within the same sentence'
        ents[ent_id] = {
            'id': ent_id,
            'type': ent_type,
            'start': int(start),
            'end': int(end),
            'text': text
        }
    return results

# {article_id: [{relation}]}
def read_rel_file(filename):
    results = {}
    lines = utils.read_tsv(filename)
    for line in lines:

        assert len(line) == 4, 'each line of relation files should have 4 columns'
        article_id, rel_type, ent_id1, ent_id2 = line
        if args.debug is not None and article_id != args.debug:
            continue

        # process relation
        ent_id1, ent_id2 = ent_id1.split(':')[1], ent_id2.split(':')[1]
        if article_id not in results:
            results[article_id] = []
        rels = results[article_id]
        rels.append({
            'ent_id1': ent_id1,
            'ent_id2': ent_id2,
            'rel_type': rel_type
        })
    return results

# {article_id: {title: str, sentences(sorted): [{sentence information}]}}
def read_abs_file(filename, verbose=False):
    results = {}
    lines = utils.read_tsv(filename)
    tokenizer = utils.get_tokenizer()

    # enable tqdm
    if verbose:
        lines = tqdm(lines, desc='Splitting abstract into sentences')
    for i, line in enumerate(lines):

        assert len(line) == 3, 'each line of relation files should have 3 columns'
        article_id, title, abstract = line
        if args.debug is not None and article_id != args.debug:
            continue
        abstract = ' '.join([title, abstract]) # the span/re started from title

        # sentencizing abstract
        sents = [{
            'start': s.start_char,
            'end': s.end_char,
            'text': s.text
        } for s in list(tokenizer(abstract).sents)]

        assert article_id not in results, 'article id should be unique'
        results[article_id] = {
            'abstract': sents
        }
    return results

# assume both span are sorted and s1 cannot span across s2
# if assumption is violated, we assume the tokenizer made an mistake
# also s1 cannot go beyond the very last s2, this can never be violated
def span_insert_helper(spans1, spans2, insert_key):
    i1 = 0
    i2 = 0
    failed = [] # keep track of mistakes tokenizer made
    # keep trying to insert s1 into s2
    # if s1 is out of bound, try next s2
    while i1 < len(spans1):
        s1 = spans1[i1]
        s2 = spans2[i2]
        if s1['start'] >= s2['end']:
            i2 += 1
            assert i2 < len(spans2), 's1 cannot go beyond last s2'
            continue
        if s1['end'] <= s2['end']:
            s2[insert_key].append(s1)
            i1 += 1
            continue
        failed.append((s1, s2))
        i1 += 1
    return failed

def process_dataset(ent_filename, rel_filename, abs_filename, verbose=False):
    art_ent_dict = read_ent_file(ent_filename)
    art_rel_dict = read_rel_file(rel_filename)
    art_abs_dict = read_abs_file(abs_filename, verbose)

    # go through all relations, retrive information from entities dictionary
    for article_id, rels in art_rel_dict.items():
        ents = art_ent_dict[article_id]
        for rel in rels:
            ent1 = ents[rel['ent_id1']]
            ent2 = ents[rel['ent_id2']]

            # calculate relation span
            rel['start'] = min(ent1['start'], ent2['start'])
            rel['end'] = max(ent1['end'], ent2['end'])
    
    # go through each abstract, and put entities and relations along with
    # the correct sentence. aka:
    # {article_id: {title: str, sentences(sorted): [{text, span, entities, relations}]}}
    for article_id, article in art_abs_dict.items():

        # get relation and entities related to this article
        rels = art_rel_dict.get(article_id, [])
        ents = art_ent_dict.get(article_id, {})
        ents = [e for e in ents.values()] # dict to list

        # sort the relation and entitiy span to help the next step
        rels.sort(key=lambda x: (x['start'], x['end']))
        ents.sort(key=lambda x: x['start'])

        # initialize article sentence 
        for sent in article['abstract']:
            sent['relations'] = []
            sent['entities'] = []

        # span insert can fail because scispacy can mistakenly split a single sentence into two
        failed_rel = span_insert_helper(rels, article['abstract'], 'relations')
        failed_ent = span_insert_helper(ents, article['abstract'], 'entities')
        if len(failed_rel) != 0:
            print(f'WARNING: article_id = {article_id}, {len(failed_rel)} relations are ignored because they span across multiple sentences')
        if len(failed_ent) != 0:
            print(f'WARNING: article_id = {article_id}, {len(failed_ent)} entities are ignored because they span across multiple sentences')
    
    return art_abs_dict

def main(ent_file, rel_file, abs_file, out_file, verbose=False):
    processed_data = process_dataset(ent_file, rel_file, abs_file, verbose)
    utils.save_json(out_file, processed_data)

if __name__ == '__main__':

    # converting original datasets into json format

    parser = argparse.ArgumentParser()
    parser.add_argument('ent_file')
    parser.add_argument('rel_file')
    parser.add_argument('abs_file')
    parser.add_argument('-o', '--out_file', required=True)
    parser.add_argument('-d', '--debug')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args.ent_file, args.rel_file, args.abs_file, args.out_file, args.verbose)