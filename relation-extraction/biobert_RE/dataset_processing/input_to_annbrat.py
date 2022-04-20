from utils import utils
from utils import json_to_brat
import argparse
import os
from tqdm import tqdm

def convert_json_to_brat(json_file, ann_folder):
    dataset = utils.read_json(json_file)
    for articleId, data in dataset.items():
        txt, ann = json_to_brat.article_brat_repr(data, include_entities=True)
        with open(f'{ann_folder}/{articleId}.ann', 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(ann))

def input_to_re(json_file, input_file, ann_folder):
    rel_id = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    json_file = utils.read_json(json_file)

    for line in tqdm(data):
    
        input_id, cls, _, _ = line.split('\t')
        input_breakdown = input_id.split('_')
        
        if len(input_breakdown) == 3: # case of there is a relation
            article_id, sentence_num, rel_num = input_breakdown

            rel = json_file[article_id]['abstract'][int(sentence_num)]['relations'][int(rel_num)]
            rel_type = cls
            
            # write the relations to the ann file
            ann_file_to_append = os.path.join(ann_folder, article_id + '.ann')

            with open(ann_file_to_append, 'a+', encoding="utf8") as f:
                f.seek(0,0)
                prev_re_id = f.readlines()[-1].split('\t')[0]
                if prev_re_id[0] != 'R': # no relation recorded yet
                    rel_id = 0
                else:
                    rel_id = int(prev_re_id[1:])+1
                    
                re_to_write = f'R{rel_id}\t{rel_type} Arg1:{rel["ent_id1"]} Arg2:{rel["ent_id2"]}'
                f.writelines('\n'+re_to_write)
            
        
        
        elif len(input_breakdown) == 4: # case of NOT relation
            
            article_id, sentence_num, ent1, ent2 = input_breakdown
            

            ent1 = json_file[article_id]['abstract'][int(sentence_num)]['entities'][int(ent1)]['id']
            ent2 = json_file[article_id]['abstract'][int(sentence_num)]['entities'][int(ent2)]['id']
            rel_type = 'NOT'
            
            # write the relations to the ann file
            ann_file_to_append = os.path.join(ann_folder, article_id + '.ann')

            with open(ann_file_to_append, 'a+', encoding='utf-8') as f:
                f.seek(0,0)
                prev_re_id = f.readlines()[-1].split('\t')[0]
                if prev_re_id[0] != 'R': # no relation recorded yet
                    rel_id = 0
                else:
                    rel_id = int(prev_re_id[1:])+1
                    
                re_to_write = f'R{rel_id}\t{rel_type} Arg1:{ent1} Arg2:{ent2}'
                f.writelines('\n'+re_to_write)


if __name__ == '__main__':
    '''
    Command do turn validation.txt input to validation_gold_standard:
    python biobert_RE\dataset_processing\input_to_annbrat.py data\merged\training_original\merged.json data\merged\training\vali.txt data\merged\vali_gold
    
    Turn dev.txt input to dev_gold_standard:
    python biobert_RE\dataset_processing\input_to_annbrat.py data\merged\dev\merged.json data\merged\dev\merged.txt data\merged\dev_gold
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('json_filename')
    parser.add_argument('input_file')
    parser.add_argument('ann_folder')
    args = parser.parse_args()
    convert_json_to_brat(args.json_filename, args.ann_folder)
    input_to_re(args.json_filename, args.input_file, args.ann_folder)