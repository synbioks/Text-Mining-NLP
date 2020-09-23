import os
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="binary", help='default:binary, possible other options:{chemprot}')
args = parser.parse_args()


prob_fp = Path(args.output_path)
prob = np.loadtxt(prob_fp)



# binary
if args.task == "binary":
    pass
# chemprot
# micro-average of 5 target classes
# see "Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction (Mehryary, 2018)" for details

if args.task == "chemprot":
    indices = np.argmax(prob,axis=1)        

for i in indices:
    print(i)
