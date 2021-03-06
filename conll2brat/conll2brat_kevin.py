"""
	-> conll2brat for generating brat from huner (refined) predictions
	-> assumes that the current folder contains the folder "text" generated by the file gen_text
	   and a file "kevin.tsv" which contains wpi annotations
	-> outputs a dir called wpi_mentions which has wpi annotations in brat format
"""

import os
import pickle
import sys
from collections import defaultdict
import re
import math
import portion as P

data_dir = "./text"
out_dir = "./wpi_mentions"

def process_special_mention(fname, ent):
	return None
	# [start_offsets, len(mention), mention]

def get_offsets(fname, ents):
	with open(os.path.join(data_dir, fname), "r") as fp:
		text = fp.read()

	misses = 0
	ent_dict = defaultdict(list)
	for id, ent in enumerate(ents):
		if ent in text:
			try:
				start_offsets = [m.start() for m in re.finditer(ent, text)]
				ent_dict[id] = [start_offsets, len(ent), ent]
			# catch unbalanced paranthesis error
			except re.error:
				ent = ent[:-1]
				start_offsets = [m.start() for m in re.finditer(ent, text)]
				ent_dict[id] = [start_offsets, len(ent), ent]
		else:
			ent_dict[id] = process_special_mention(fname, ent)
			misses += 1

	# print(misses)
	return ent_dict

import pandas as pd
df = pd.read_csv('../kevin.tsv', sep='\t', header=0)

entities = {}
for model in df.columns:
	ents = []
	for ent in df[model].dropna():
		if ent is not None:
			ents.append(ent.strip())
	entities[model] = ents

# print(entities)

def process_data(fname):
	file_name = fname[:-4]
	with open(os.path.join("huner_mentions_refined", file_name+".ann"), "r") as fp:
		lines = fp.readlines()
		if(len(lines) > 0):
			last_id = int(lines[-1].strip().split("\t")[0][1:]) + 1
		else:
			# return
			last_id = 0

	ner = {}

	for line in lines:
		line_split = line.strip().split("\t")
		ent_id = line_split[0].strip()
		ent, start_ind, end_ind = line_split[1].strip().split(" ")
		mention = line_split[2].strip()

		ner[(start_ind, end_ind)] = (ent_id, ent, mention)

	# print(ner)
	# print( (33819, 33826) in ner)
	# return
	real_id = last_id
	f = open(os.path.join(out_dir, file_name+".ann"), "w")
	for model, ents in entities.items():
		ent_dict = get_offsets(fname, ents)
		
		for id, val in ent_dict.items():
			if(val is None):
				continue
			start_indices, mention_length, mention = val
			# if(mention=="E. coli"):
				# print(start_indices, start_indices[1] + mention_length)
			for start_ind in start_indices:
				end_ind = start_ind + mention_length
				if((str(start_ind), str(end_ind)) in ner):
					info = ner[(str(start_ind), str(end_ind))]
					# print(start_ind, end_ind, info)
					huner_ent = info[1]
					if(model in huner_ent):
						continue
				f.write("T{0}\t{1}_wpi {2} {3}\t{4}\n".format(real_id, model, start_ind, end_ind, mention))
				real_id += 1
	f.close()


	# solving the GFP/EGFP type of cases
	with open(os.path.join(out_dir, file_name+".ann"), "r") as fp2:
		lines2 = fp2.readlines()
	ner2 = {}

	for line in lines2:
		line_split = line.strip().split("\t")
		ent_id = line_split[0].strip()
		ent, start_ind, end_ind = line_split[1].strip().split(" ")
		mention = line_split[2].strip()

		ner2[P.closed(int(start_ind), int(end_ind))] = (ent_id, ent, start_ind, end_ind, mention)
	

	for (start_ind, end_ind), val in ner.items():
		ent_id, ent, mention = val
		ner2[P.closed(int(start_ind), int(end_ind))] = (ent_id, ent, int(start_ind), int(end_ind), mention)

	f2 = open(os.path.join(out_dir, file_name+".ann"), "w")
	# remove sub-intervals
	new_id = last_id
	for interval_small, info_small in ner2.items():
		if(info_small[1].split("_")[1]=="huner"):
			continue
		# if this interval is contained within some larger interval
		# do not write it to file
		flag = True
		for interval_large, info_large in ner2.items():
			if(interval_small == interval_large):
				continue
			if interval_large.contains(interval_small):
				ent_small = info_small[1].split("_")[0]
				ent_large = info_large[1].split("_")[0]
				if(ent_small == ent_large):
					# print(info_small, info_large)
					flag = False
					break


		# else write it to file
		if(flag):
			# print(interval_small, info)
			g1, g2, g3, g4, g5 = info_small
			f2.write("T{0}\t{1} {2} {3}\t{4}\n".format(new_id, g2, g3, g4, g5))
			new_id += 1

	f2.close()


all_files = os.listdir(data_dir)

for fname in all_files:
	if(".DS" in fname):
		continue
	# if("sb3000723" not in fname):
	# 	continue
	process_data(fname)

# my_set = set()

# for fname in os.listdir(out_dir):
# 	if(".DS" in fname):
# 		continue
# 	file_name = fname[:-4] + ".ann"
# 	with open(os.path.join(out_dir, file_name), "r") as fp:
# 		lines = fp.readlines()

# 	for line in lines:
# 		line_split = line.strip().split("\t")
# 		my_set.add(line_split[-1])

# print(len(my_set))



