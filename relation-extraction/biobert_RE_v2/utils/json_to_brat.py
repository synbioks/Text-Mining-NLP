import os

def ent_brat_repr(ent):
    return f'{ent["id"]}\t{ent["type"]} {ent["start"]} {ent["end"]}\t{ent["text"]}'

def rel_brat_repr(rel, rel_id, type_suffix=''):
    rel_type = rel['rel_type'] + type_suffix
    return f'R{rel_id}\t{rel_type} Arg1:{rel["ent_id1"]} Arg2:{rel["ent_id2"]}'

def article_brat_repr(article, include_entities=False, include_relations=False):
    txt = []
    ann = []
    for sent in article['abstract']:
        txt.append(sent['text'])
        if include_entities:
            for ent in sent['entities']:
                ann.append(ent_brat_repr(ent))
        if include_relations:
            for rel in sent['relations']:
                ann.append(rel_brat_repr(rel))
    return txt, ann

def write_brat(article_id, path, txt, ann):
    txt_filename = os.path.join(path, article_id + '.txt')
    ann_filename = os.path.join(path, article_id + '.ann')
    with open(txt_filename, 'w', encoding='utf-8') as fout:
        fout.write(' '.join(txt))
    with open(ann_filename, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(ann))
