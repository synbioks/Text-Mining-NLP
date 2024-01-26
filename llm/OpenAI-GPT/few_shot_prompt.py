import random
import re

NUM_EXAMPLES = 5

raw_prompt = "Pretend to be an expert on relation extraction, particularly for biomedical text. The following is a biomedical abstract:\n\n\"{}\"\n\nI need to categorize each pairwise chemical name (pair of entities) in the abstract as having one of the following relational classes (ChemProt Relations):\n\nCPR-1: 'PART_OF'\nCPR-2: 'REGULATOR|DIRECT_REGULATOR|INDIRECT_REGULATOR'\nCPR-3: 'UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR'\nCPR-4: 'DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR'\nCPR-5: 'AGONIST|AGONIST-ACTIVATOR|AGONIST_INHIBITOR'\nCPR-6: 'ANTAGONIST'\nCPR-7: 'MODULATOR|MODULATOR_ACTIVATOR|MODULATOR_INHIBITOR'\nCPR-8: 'COFACTOR'\nCPR-9: 'SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF'\nCPR-10: 'NOT'\n\nCan you list each pair of entities that you find in the abstract and their relational class in the format (entity_1, entity_2, CPR_class), along with an explanation as to why that particular class was chosen?\n\nTo help you with this task, the following is a list of identified chemicals/genes from the abstract -- their identifier, whether it is a chemical or gene, line numbers in which they would be found in the text, and their name. Use these chemicals/genes to form your relations and their categories:\n\n{}\n\nFor example, {} would be a non-exhaustive list of some example relationships that you will find."

abstract_file = "/sbksvol/amurali/llm/11488610.txt"

with open(abstract_file, "r") as f:
    abstract = f.read()

ann_file = "/sbksvol/amurali/llm/11488610.ann"
entities = ""
example_list = []
with open(ann_file, "r", encoding="utf8") as f:
    for line in f:
        if(line.startswith("T") and len(line) > 0):
            entities += line
        if(line.startswith("R") and len(line) > 0):
            example_list.append(line)

random_examples = random.sample(example_list, NUM_EXAMPLES)
random_examples_str = ' '.join(random_examples)

entity_regex = r"\bT\d+\b"
class_regex = r"\bCPR-\b\d+|\bNOT\b"

entity_list = re.findall(entity_regex, random_examples_str)
class_list = re.findall(class_regex, random_examples_str)

num_classes = len(class_list)

examples = ""

for i in range(num_classes):
    if(i == num_classes - 1):
        examples += "({}, {}, {})".format(entity_list[i*2], entity_list[i*2 + 1], class_list[i])
    else:
        examples += "({}, {}, {}), ".format(entity_list[i*2], entity_list[i*2 + 1], class_list[i])

prompt = raw_prompt.format(abstract, entities, examples)

print(prompt)
