"""
	handle the ? and FIG tokens from ACS_huner
	assumes that the current folder contains the folders
	'chemical', 'gene', 'species' and 'cellline' with this script
	See https://github.com/hu-ner/huner/issues/19 for an example
"""

import os
import sys
import re

models = ["chemical", "gene", "species", "cellline"]

files = os.listdir("chemical")

if (".DS_Store" in files):
	files.remove(".DS_Store")


def process_file(model, fname):
	# create refined conll version
	with open(os.path.join(model, fname), "r") as fp:
		lines = fp.readlines()

	f_data = open(os.path.join(model+"_data", fname), "w")

	for line in lines:
		if(line=="\n"):
			f_data.write(line)
			continue
		line_split = line.split("\t")
		token = line_split[0].strip()
		tag = line_split[2].strip()
		if(len(tag)>1):
			tag = tag[0]
		conf = line_split[3].strip()
		if(token == "FIG" or token == "?"):
			# print("Buenos di'as", model, " ", fname)
			continue
		elif("?" in token):
			# print("iHola!", model, " ", fname)
			token = re.sub("\?", "", token)

		f_data.write(token+"\t"+tag+"\t"+conf+"\n")

	f_data.close()

for model in models:
	try:
		os.mkdir(model+"_data")
	except:
		pass

	for fname in files:
		process_file(model, fname)

