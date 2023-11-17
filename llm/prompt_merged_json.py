import json
import random

NUM_EXAMPLES = 2
PURE_ONE_SHOT = False #True
ZERO_SHOT = False

f = open('../data/merged.json')
corpus = json.load(f)
f.close()


#Generate one and few shot example(s)
examples_dict = {}

#examples_dict format: {'CPR-1': [], 'CPR-2': [], 'CPR-3': [], 'CPR-4': [], 'CPR-5': [], ..., 'NOT': []}
for i in range(1, 10):
    examples_dict['CPR-{}'.format(i)] = []

examples_dict['NOT'] = []

#pure one-shot (one example total)
random_example = None

if(NUM_EXAMPLES == 1 and PURE_ONE_SHOT):
    #find a sentence with explicit relations; use first relation as one-shot example
    #example in the form (ent_id1, ent_id2, rel_type, sentence text)
    while(random_example is None):
        random_abstract = random.choice(list(corpus.keys()))
        for sentence in corpus[random_abstract]['abstract']:
            if(len(sentence['relations']) >= 1):
                random_example = (sentence['relations'][0]['ent_id1'], sentence['relations'][0]['ent_id2'], sentence['relations'][0]['rel_type'], sentence['text'])
                break


#one-shot with one example per class, or few-shot with a few examples per class
if(NUM_EXAMPLES >= 1 and not PURE_ONE_SHOT):
    #find examples for each CPR class until you have NUM_EXAMPLES examples per class
    for abstract in corpus:
        for sentence in corpus[abstract]['abstract']:
            if(len(sentence['relations']) >= 1):
                for relation in sentence['relations']:
                    if(len(examples_dict[relation['rel_type']]) < NUM_EXAMPLES):
                        examples_dict[relation['rel_type']].append((relation['ent_id1'], relation['ent_id2'], relation['rel_type'], sentence['text']))
            else:
                if(len(sentence['entities']) > 1):
                    if(len(examples_dict['NOT']) < NUM_EXAMPLES):
                        entity_pairs = [(entity_one['id'], entity_two['id']) for idx, entity_one in enumerate(sentence['entities']) for entity_two in sentence['entities'][idx + 1:]]
                        pairs_index = 0
                        while(len(examples_dict['NOT']) < NUM_EXAMPLES and pairs_index < len(entity_pairs)):
                            examples_dict['NOT'].append((entity_pairs[pairs_index][0], entity_pairs[pairs_index][1], 'NOT', sentence['text']))
                            pairs_index += 1

#print(random_example)
print(examples_dict)


raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the sentence as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the sentence and their relational class in the format (entity_1, entity_2, CPR_class), along with an explanation as to why that particular class was chosen?\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence -- their identifier ('id'), whether it is a chemical or gene ('type'), line numbers in which they would be found in the full abstract ('start' and 'end'), and their name ('text'). Use these chemicals/genes to form your relations and their categories:\n\n{}\n\nFor example, {} would be a non-exhaustive list from the entire corpus of abstracts of some example relationships for each class."

if(ZERO_SHOT):
    raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the sentence as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the sentence and their relational class in the format (entity_1, entity_2, CPR_class), along with an explanation as to why that particular class was chosen?\n\nTo help you with this task, the following is a list of identified chemicals/genes from the sentence -- their identifier ('id'), whether it is a chemical or gene ('type'), line numbers in which they would be found in the full abstract ('start' and 'end'), and their name ('text'). Use these chemicals/genes to form your relations and their categories:\n\n{}"

#abstract = corpus['10064839']['abstract']
#
#for sentence in abstract:
#    entities = sentence['entities']
#    relations = sentence['relations']
#
#    prompt_relations = ""
#    prompt_entities = ""
#
#    for i in entities:
#        prompt_entities += str(i) + '\n'
#
#    if(len(relations) < 1):
#        if(len(entities) <= 1):
#            #zero_shot = True
#            continue
#        random_entity_pairs = [(entity_one['id'], entity_two['id']) for idx, entity_one in enumerate(entities) for entity_two in entities[idx + 1:]]
#        random_relations = random.sample(random_entity_pairs, min(len(random_entity_pairs), NUM_EXAMPLES))
#        example_relations = [(i[0], i[1], 'NOT') for i in random_relations]
#
#        for i in range(len(example_relations)):
#            if(i < len(example_relations) - 1):
#                prompt_relations += str(example_relations[i]) + ", "
#            else:
#                prompt_relations += str(example_relations[i])
#    else:
#        random_relations = random.sample(relations, min(len(relations), NUM_EXAMPLES))
#        example_relations = [(i['ent_id1'], i['ent_id2'], i['rel_type']) for i in random_relations]
#
#        for i in range(len(example_relations)):
#            if(i < len(example_relations) - 1):
#                prompt_relations += str(example_relations[i]) + ", "
#            else:
#                prompt_relations += str(example_relations[i])
#    
#    #zero_shot = random.random() < 0.1
#
#    #if(zero_shot):
#    #    raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the sentence as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the sentence and their relational class in the format (entity_1, entity_2, CPR_class), along with an explanation as to why that particular class was chosen?\n\nTo help you with this task, the following is a list of identified chemicals/genes from the sentence -- their identifier ('id'), whether it is a chemical or gene ('type'), line numbers in which they would be found in the full abstract ('start' and 'end'), and their name ('text'). Use these chemicals/genes to form your relations and their categories:\n\n{}"
#    prompt = raw_prompt.format(sentence['text'], prompt_entities, prompt_relations)
#    print(prompt)
#
