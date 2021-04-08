"""
	copy the 100 documents listed in select100.txt from dir combined_mentions to dir select100
"""
import os
import sys
from collections import defaultdict
import shutil

data_dir = "combined_mentions"
# out_dir = "transfer"
out_dir = "select100"

data_dict = {}

files = os.listdir(data_dir)
try:
	files.remove(".DS_Store")
except:
	pass

with open("select100.txt", "r") as fp:
	docs = fp.readlines()

def process_file(doc):
	f = open(os.path.join(data_dir, doc+".ann"), "r")
	f_write = open(os.path.join(out_dir, doc+".ann"), "w")

	lines = f.readlines()

	data_set = {}

	for line in lines:
		line_split = line.strip().split("\t")
		entity_type = line_split[1].strip().split(" ")[0]
		if(len(line_split) < 3):
			continue

		mention = line_split[2].strip()
		if(entity_type[-5:]=="huner"):
			f_write.write(line)
			data_set[mention] = entity_type
		elif(entity_type[-3:]=="wpi"):
			if(mention in data_set and data_set[mention][:-5]==entity_type[:-3]):
				continue
			else:
				f_write.write(line)

	f_write.close()
	f.close()

for doc in docs:
	process_file(doc.strip())
	doc = doc.strip()
	source_f = os.path.join("text", doc+".txt")
	dest_f = os.path.join(out_dir, doc+".txt")
	shutil.copyfile(source_f, dest_f)

# # data_dir = "select100"
# data_dir = "combined_mentions"
# out_dir = "transfer2"

# files = os.listdir(data_dir)

# for fname in files:
# 	if(".DS_Store" in fname or ".conf" in fname):
# 		continue
# 	else:
# 		dir_name = fname[:3]
# 		try:
# 			os.mkdir(os.path.join(out_dir, dir_name))
# 		except:
# 			pass
# 		source_f = os.path.join(data_dir, fname)
# 		dest_f = os.path.join(out_dir, dir_name, fname)
# 		shutil.copyfile(source_f, dest_f)
