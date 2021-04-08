"""
	combine the brat annotations for huner and wpi's new mentions
"""
import os
import sys

huner_dir = './huner_mentions_refined'
wpi_dir = './wpi_mentions'
comb_dir = './combined_mentions'

files = os.listdir(wpi_dir)
print(len(files))

# files_wpi = os.listdir(wpi_dir)
# files_huner = os.listdir(huner_dir)
# for fname in files_wpi:


for fname in files:
	if(".DS_Store" in fname):
		continue

	f = open(os.path.join(comb_dir, fname), "w")

	huner_file = open(os.path.join(huner_dir, fname), "r")
	wpi_file = open(os.path.join(wpi_dir, fname), "r")

	huner_lines = huner_file.readlines()
	wpi_lines = wpi_file.readlines()

	for line_huner in huner_lines:
		f.write(line_huner)

	for line_wpi in wpi_lines:
		f.write(line_wpi)

	huner_file.close()
	wpi_file.close()

	f.close()