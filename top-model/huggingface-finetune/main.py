from run import main
import argparse
import os
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--seed_value", default=42)
parser.add_argument("--set_seed", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--entity_type", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--exp_name", required=True)
parser.add_argument("--exp_config", required=True)
parser.add_argument("--root", required=True)
args = parser.parse_args()

# code directory
CODE_DIR = '/sbks-ucsd/top-model/huggingface-finetune/'

# exp config file
EXP_CONFIG_FILE = os.path.join(args.root, CODE_DIR, 'exp_config.json')

with open(EXP_CONFIG_FILE, 'r') as fp:
    params = json.load(fp)[args.exp_config]

params['seed_value'] = args.seed_value
params['set_seed'] = args.set_seed
params['entity_type'] = args.entity_type
params['dataset'] = args.dataset
params['exp_name'] = args.exp_name
params['exp_config'] = args.exp_config

# All file paths have to be absolute paths
params['WORKING_DIR'] = os.path.join(
    params['WORKING_DIR'], args.exp_name)
params['DATA_PATH'] = os.path.join(args.data, "NER_data/")
params['CACHE_DIR'] = os.path.join(
    params['WORKING_DIR'], "NER_out_test")

# Where model checkpoints are stored.
params['OUTPUT_DIR'] = os.path.join(
    params['WORKING_DIR'], "model_output_test/")
params['TRAIN_ARGS_FILE'] = os.path.join(
    params['WORKING_DIR'], "train_args_test.json")

shutil.rmtree(params['WORKING_DIR'])
try:
    os.makedirs(params['WORKING_DIR'])
except OSError as e:
    print(
        f'Dir - {params["WORKING_DIR"]} - {e.os.strerror} . It mean same directory being used for multiple exp')
main(params)
