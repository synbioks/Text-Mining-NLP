"""
	-> create text files out of conll files
	-> assumes that the current folder contains the folders
	'chemical_data' with this script which can be created using the file refine_conll.py
	-> creates a folder called "text" and places all ACS text articles into it.

"""
import os
import sys
import re

files = os.listdir("chemical_data")

if (".DS_Store" in files):
	files.remove(".DS_Store")


def process_file(model, fname):
	
	with open(os.path.join(model, fname), "r") as fp:
		lines = fp.readlines()

	f_text = open(os.path.join("text", fname[:-5] + "txt"), "w")

	for line in lines:
		if(line=="\n"):
			f_text.write(line)
			continue
		line_split = line.split("\t")
		token = line_split[0].strip()

		f_text.write(token+" ")

	f_text.close()

try:
	os.mkdir("text")
except:
	pass

for fname in files:
	process_file("chemical_data", fname)

