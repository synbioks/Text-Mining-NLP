"""
	use the confidence scores to generate a unique prediction for each mention
"""
import os
import sys
import portion as P

from collections import defaultdict

models = ["chemical", "gene", "species", "cellline"]
data_dir = "huner_mentions"
out_dir = "huner_mentions_refined"

files = os.listdir(data_dir)
if(".DS_Store" in files):
	files.remove(".DS_Store")

def process_file(fname):
	with open(os.path.join(data_dir, fname), "r") as fp:
		lines = fp.readlines()

	f_out = open(os.path.join(out_dir, fname), "w")

	ner = defaultdict(list)

	for line in lines:
		line_split = line.strip().split("\t")
		ent_id = line_split[0].strip()
		ent, start_ind, end_ind = line_split[1].strip().split(" ")
		mention = line_split[2].strip()
		conf = line_split[3].strip()

		ner[(start_ind, end_ind)].append((ent_id, ent, mention, conf))

	# for equal intervals, pick one with max confidence
	ner2 = {}
	max_conf = -1; opt_conf_info = None
	for (start_ind, end_ind), info in ner.items():
		max_conf = -1; opt_conf_info = None
		for (ent_id, ent, mention, conf) in info:
			if(float(conf) > max_conf):
				max_conf = float(conf)
				opt_conf_info = (ent_id, ent, start_ind, end_ind, mention)

		g1, g2, g3, g4, g5 = opt_conf_info
		ner2[P.closed(int(g3), int(g4))] = (g1, g2, g3, g4, g5)

	# remove sub-intervals
	new_id = 0
	for interval_small, info in ner2.items():
		# if this interval is contained within some larger interval
		# do not write it to file
		flag = True
		for interval_large in ner2:
			if(interval_small == interval_large):
				continue
			if interval_large.contains(interval_small):
				# print(interval_large, interval_small, info)
				flag = False
				break

		# else write it to file
		if(flag):
			g1, g2, g3, g4, g5 = info
			f_out.write("T{0}\t{1} {2} {3}\t{4}\n".format(new_id, g2, g3, g4, g5))
			new_id += 1

	# new_id = 0
	# for start_ind, info in ner.items():
	# 	# find maximum by length
	# 	max_len = -1; opt_len_info = []
	# 	for (ent_id, ent, end_ind, mention, conf) in info:
	# 		if((int(end_ind)-int(start_ind)) >= max_len):
	# 			max_len = int(end_ind)-int(start_ind)
	# 			opt_len_info.append((ent_id, ent, start_ind, end_ind, mention, conf))

	# 	# if there are multiple maximum's by length, find max by confidence score
		# max_conf = -1; opt_conf_info = None
		# for (ent_id, ent, start_ind, end_ind, mention, conf) in opt_len_info:
		# 	if(float(conf) > max_conf):
		# 		max_conf = float(conf)
		# 		opt_conf_info = (ent_id, ent, start_ind, end_ind, mention)

	# 	g1, g2, g3, g4, g5 = opt_conf_info
	# 	f_out.write("T{0}\t{1} {2} {3}\t{4}\n".format(new_id, g2, g3, g4, g5))
	# 	new_id += 1r

	f_out.close()

for fname in files:
	# if(fname != "sb3000673.ann"):
	# 	continue
	process_file(fname)