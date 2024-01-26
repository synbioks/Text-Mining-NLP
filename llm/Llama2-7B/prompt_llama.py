import json
import random
import subprocess

NUM_EXAMPLES = 2
PURE_ONE_SHOT = False

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
                entity_id_one = sentence['relations'][0]['ent_id1']
                entity_id_two = sentence['relations'][0]['ent_id2']
                entity_name_one = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_one][0]
                entity_name_two = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_two][0]
                random_example = (entity_name_one, entity_name_two, sentence['relations'][0]['rel_type'], sentence['text'])
                break


#one-shot with one example per class, or few-shot with a few examples per class
if(NUM_EXAMPLES >= 1 and not PURE_ONE_SHOT):
    #find examples for each CPR class until you have NUM_EXAMPLES examples per class
    for abstract in corpus:
        for sentence in corpus[abstract]['abstract']:
            if(len(sentence['relations']) >= 1):
                for relation in sentence['relations']:
                    if(len(examples_dict[relation['rel_type']]) < NUM_EXAMPLES):
                        entity_id_one = relation['ent_id1']
                        entity_id_two = relation['ent_id2']
                        entity_name_one = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_one][0]
                        entity_name_two = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_two][0]
                        examples_dict[relation['rel_type']].append((entity_name_one, entity_name_two, relation['rel_type'], sentence['text']))
            else:
                if(len(sentence['entities']) > 1):
                    if(len(examples_dict['NOT']) < NUM_EXAMPLES):
                        #entity_pairs = [(entity_one['id'], entity_two['id']) for idx, entity_one in enumerate(sentence['entities']) for entity_two in sentence['entities'][idx + 1:]]
                        entity_pairs = [(entity_one['text'], entity_two['text']) for idx, entity_one in enumerate(sentence['entities']) for entity_two in sentence['entities'][idx + 1:]]
                        pairs_index = 0
                        while(len(examples_dict['NOT']) < NUM_EXAMPLES and pairs_index < len(entity_pairs)):
                            examples_dict['NOT'].append((entity_pairs[pairs_index][0], entity_pairs[pairs_index][1], 'NOT', sentence['text']))
                            pairs_index += 1

example_sentence = corpus['10064839']['abstract'][0]

raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the sentence as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the sentence and their relational class in the format (entity_1, entity_2, CPR_class)?\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence -- their identifier ('id'), whether it is a chemical or gene ('type'), line numbers in which they would be found in the full abstract ('start' and 'end'), and their name ('text'). Use these chemicals/genes to form your relations and their categories:\n\n{}\n\nFurthermore, some example relations for each class (randomly selected from the entire corpus of abstracts) are provided here, in the form, (entity_1, entity_2, CPR_class, sentence in which they are found):\n\n{}\n\nPlease use these examples as calibration to help you with this task."

examples_string = ''
for key, val in examples_dict.items():
    examples_string += key + ':\n'
    for example in val:
        examples_string += str(example) + '\n'

prompt = raw_prompt.format(example_sentence['text'], example_sentence['entities'], examples_string)
#prompt = "What is Llama the llm?"

if(NUM_EXAMPLES == 0):
    raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the sentence as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the sentence and their relational class in the format (entity_1, entity_2, CPR_class)?\n\nTo help you with this task, the following is a list of identified chemicals/genes from the sentence -- their identifier ('id'), whether it is a chemical or gene ('type'), line numbers in which they would be found in the full abstract ('start' and 'end'), and their name ('text'). Use these chemicals/genes to form your relations and their categories:\n\n{}"
    prompt = raw_prompt.format(example_sentence['text'], example_sentence['entities'])


from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

print("Pytorch Version: {}".format(torch.__version__))

#model = "./Llama-2-7b-hf/"
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
#huggingface-cli login
#tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

#llama_pipeline = pipeline(
#    "text-generation",
#    model=model,
#    torch_dtype=torch.float16,
#    device_map="auto"
#)

llama_pipeline = pipeline(
    "conversational",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

def get_llama_response(prompt: str):
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        sequences[0]['generated_text']: The model's response.
    """

    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096
        #max_length=100#,
        )
    return sequences[0]['generated_text']

print(get_llama_response(prompt))
