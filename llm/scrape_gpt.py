import requests
from bs4 import BeautifulSoup
import re
import csv

chat_url = "https://chat.openai.com/share/d202bc00-1c69-4aae-8f5c-a17eaae7631d" #zero-shot
#chat_url = "https://chat.openai.com/share/dfa6a88c-1c3b-4dfd-9261-998b49fdc493" #one-shot
#chat_url = "https://chat.openai.com/share/06c3e386-0343-4e20-8059-eabdd4f0d90c" #few-shot

zero_shot = True

response = requests.get(chat_url)
soup = BeautifulSoup(response.text, 'html.parser')
gpt_output = soup.find_all('li')

results = []

for li_tag in gpt_output:
    p_tags = li_tag.find_all('p')
    for p_tag in p_tags:
        text = p_tag.text
        #regex = r"\(.+\,.+\,.+\)"
        regex = r"\([^,\)]+\,[^,\)]+\,[^,\)]+\)"
        classification = re.findall(regex, text)[0]
        results.append(classification)

if(zero_shot):
    entity_regex = r"\bT\d+\b"
    class_regex = r"\bCPR-\b\d+|\bNOT\b"
    for i in range(len(results)):
        entity_list = re.findall(entity_regex, results[i])
        class_list = re.findall(class_regex, results[i])

        if(class_list[0] == 'NOT'): class_list[0] = 'CPR-10'

        results[i] = '{}'.format(entity_list[0]) + '\t' + '{}'.format(entity_list[1]) + '\t' +  '{}'.format(class_list[0][-1])

with open("results.tsv", "w") as f:
    f.write('id1' + '\t' + 'id2' + '\t' + 'class' + '\n')
    for i in range(len(results)):
        if(i < len(results) - 1):
            f.write(results[i] + '\n')
        else:
            f.write(results[i])
