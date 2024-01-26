import json
import random
import subprocess

NUM_EXAMPLES = 1
PURE_ONE_SHOT = False

f = open('/sbksvol/amurali/data/merged.json')
corpus = json.load(f)
f.close()


#Generate one and few shot example(s)
examples_dict = {}

#examples_dict format: {'RELATED': [], 'NOT_RELATED': []}

examples_dict['RELATED'] = []
examples_dict['NOT_RELATED'] = []

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
                relation_type = 'NOT_RELATED'
                if(sentence['relations'][0]['rel_type'] != 'NOT'):
                    relation_type = 'RELATED'
                random_example = (entity_name_one, entity_name_two, relation_type, sentence['text'])
                break


#one-shot with one example per class, or few-shot with a few examples per class
if(NUM_EXAMPLES >= 1 and not PURE_ONE_SHOT):
    #find examples for each CPR class until you have NUM_EXAMPLES examples per class
    for abstract in corpus:
        for sentence in corpus[abstract]['abstract']:
            if(len(sentence['relations']) >= 1):
                for relation in sentence['relations']:
                    relation_type = 'NOT_RELATED'
                    if(relation['rel_type'] != 'NOT'):
                        relation_type = 'RELATED'
                    if(len(examples_dict[relation_type]) < NUM_EXAMPLES):
                        entity_id_one = relation['ent_id1']
                        entity_id_two = relation['ent_id2']
                        entity_name_one = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_one][0]
                        entity_name_two = [entity_dict['text'] for entity_dict in sentence['entities'] if entity_dict['id'] == entity_id_two][0]
                        examples_dict[relation_type].append((entity_name_one, entity_name_two, relation_type, sentence['text']))
            else:
                if(len(sentence['entities']) > 1):
                    if(len(examples_dict['NOT_RELATED']) < NUM_EXAMPLES):
                        #entity_pairs = [(entity_one['id'], entity_two['id']) for idx, entity_one in enumerate(sentence['entities']) for entity_two in sentence['entities'][idx + 1:]]
                        entity_pairs = [(entity_one['text'], entity_two['text']) for idx, entity_one in enumerate(sentence['entities']) for entity_two in sentence['entities'][idx + 1:]]
                        pairs_index = 0
                        while(len(examples_dict['NOT_RELATED']) < NUM_EXAMPLES and pairs_index < len(entity_pairs)):
                            examples_dict['NOT_RELATED'].append((entity_pairs[pairs_index][0], entity_pairs[pairs_index][1], 'NOT_RELATED', sentence['text']))
                            pairs_index += 1

example_sentence = corpus['9990013']['abstract'][0]

expected_sentence = corpus['9890894']['abstract'][3]
expected_response = "Sure, here you go:\n\n1. ('sulfate', 'immunoglobulin (Ig)-like module II', 'RELATED')\n\n2. ('sulfate', 'FGFR', 'RELATED')\n\n3. ('immunoglobulin (Ig)-like module II', 'FGFR', 'NOT_RELATED')."

raw_prompt = "***THIS IS AN EXAMPLE PROMPT AND EXPECTED RESPONSE***:\n\n### Instruction:\nPretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need you to categorize each pairwise chemical name (pair of entities) in the sentence as being 'RELATED' or 'NOT_RELATED' in the format (entity_1, entity_2, class).\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence:\n{}\n\nUse these chemicals/genes to form your relations and their categories.\n\nFurthermore, some example relations for each class (randomly selected from the entire corpus of abstracts) are provided here, in the format, (entity_1, entity_2, class, example_sentence):\n\n{}\n\nPlease use these examples as calibration to help you with this task.\n\n### Response:\n{}\n\n***NOW, IT IS YOUR TURN***:\n\n### Instruction:\nPretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need you to categorize each pairwise chemical name (pair of entities) in the sentence as being 'RELATED' or 'NOT_RELATED' in the format (entity_1, entity_2, class).\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence:\n{}\n\nUse these chemicals/genes to form your relations and their categories.\n\nFurthermore, some example relations for each class (randomly selected from the entire corpus of abstracts) are provided here, in the format, (entity_1, entity_2, class, example_sentence):\n\n{}\n\nPlease use these examples as calibration to help you with this task.\n\n### Response:\nSure, here you go:\n\n"


examples_string = ''
for key, val in examples_dict.items():
    examples_string += key + ':\n'
    for example in val:
        examples_string += str(example) + '\n'

example_entities = example_sentence['entities']
entity_text = ''
for entity in example_entities:
    entity_text += '\n' + entity['text']

expected_entities = expected_sentence['entities']
expected_entity_text = ''
for entity in expected_entities:
    expected_entity_text += '\n' + entity['text']

prompt = raw_prompt.format(expected_sentence['text'], expected_entity_text, examples_string, expected_response, example_sentence['text'], entity_text, examples_string)

if(NUM_EXAMPLES == 0):
    raw_prompt = "***THIS IS AN EXAMPLE PROMPT AND EXPECTED RESPONSE***:\n\n### Instruction:\nPretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need you to categorize each pairwise chemical name (pair of entities) in the sentence as being 'RELATED' or 'NOT_RELATED' in the format (entity_1, entity_2, class).\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence:\n{}\n\nUse these chemicals/genes to form your relations and their categories.\n\n### Response:\n{}\n\n***NOW, IT IS YOUR TURN***:\n\n### Instruction:\nPretend to be an expert on relation extraction, particularly for biomedical text. The following is a sentence from a biomedical abstract:\n\n\"{}\"\n\nI need you to categorize each pairwise chemical name (pair of entities) in the sentence as being 'RELATED' or 'NOT_RELATED' in the format (entity_1, entity_2, class).\n\nTo help you with this task, the following is a list of identified chemicals/genes (entities) from the sentence:\n{}\n\nUse these chemicals/genes to form your relations and their categories.\n\n### Response:\nSure, here you go:\n\n"
    prompt = raw_prompt.format(expected_sentence['text'], expected_entity_text, expected_response, example_sentence['text'], entity_text)


from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, Conversation
import torch

print("Pytorch Version: {}".format(torch.__version__))

#model = "./Llama-2-7b-hf/"
model = "meta-llama/Llama-2-7b-chat-hf"

#tokenizer = AutoTokenizer.from_pretrained(model)
#huggingface-cli login
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

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

    conversation = Conversation(prompt)
    sequences = llama_pipeline(
        conversation,
        #prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=10000,
        max_length=10000
    )
    #return sequences[0]['generated_text']
    return sequences.generated_responses[0]

print(prompt)
print(get_llama_response(prompt))
