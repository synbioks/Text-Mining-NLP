"""
	-> conll2brat for generating brat from huner (refined) predictions
	-> assumes that the current folder contains the folders
	'chemical_data', 'gene_data', 'species_data', 'cellline_data' 
	with this script which can be created using the previous files
	-> outputs a dir called huner_mentions which contains the huner annotations
	in brat format
"""

import os
import sys

out_dir = "./huner_mentions"

models = ["chemical", "gene", "species", "cellline"]

files = os.listdir("chemical_data")

if (".DS_Store" in files):
	files.remove(".DS_Store")

def process_file(fname):
	f_out = open(os.path.join(out_dir, fname[:-5] + "ann"), "w")

	real_id = 0
	for model in models:
		# if(model != "chemical"):
			# continue

		with open(os.path.join(model+"_data", fname), "r") as fp:
			lines = fp.readlines()

		num_lines = len(lines)
		offset = 0
		i = 0
		
		while i < num_lines:
			line = lines[i]
			if(line=="\n"):
				i += 1
				offset += 1
				continue

			mention = ""

			line_split = line.split("\t")
			token = line_split[0].strip()
			tag = line_split[1].strip()
			conf = line_split[2].strip()

			mention += token

			# if tag is "B", scan next entity to see if it's an "I"...
			if(tag == "B"):
				# if(token == "N-terminal"):
				# 	print(offset)
				i += 1
				start_ind = offset
				offset += len(token) + 1
				
				# if it is, keep itearting through the file (line by line) until you keep seeing "I"
				while i< num_lines and lines[i] != "\n" and lines[i].split("\t")[1].strip() == "I":
					next_token = lines[i].split("\t")[0].strip()
					mention += " " + next_token
					offset += len(next_token) + 1
					i += 1

				end_ind = offset - 1
				# BRAT format line
				f_out.write("T{0}\t{1}_huner {2} {3}\t{4}\t{5}\n".format(real_id, model, start_ind, end_ind, mention, conf))
				real_id += 1

				i -= 1
			else:
				offset += len(token) + 1

			i += 1

	f_out.close()



for fname in files:
	# if(fname != "sb5b00002.CONLL"):
	# 	continue
	process_file(fname)

