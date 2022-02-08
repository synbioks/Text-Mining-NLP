"""
transformers==4.6.0 
pytorch-crf==0.7.2 
"ray[tune]"==1.9.2 
wandb==0.12.9
"""

from models.models_enum import ModelsType
from run_v2 import main
import argparse
import os
import json
import shutil

os.environ["WANDB_API_KEY"]="90852721fdf4fb388c7f75ad45a5a0629bfc4bbf"

parser = argparse.ArgumentParser()
parser.add_argument("--seed_value", default=42, type=int)
parser.add_argument("--set_seed", action="store_true")
parser.add_argument("--data", required=True)
parser.add_argument("--entity_type", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--exp_name", required=True)
parser.add_argument("--exp_config", required=True)
parser.add_argument("--root", required=True)
args = parser.parse_args()

# code directory
CODE_DIR = 'sbks-ucsd-test/top-model/huggingface-finetune/'

# exp config file
EXP_CONFIG_FILE = os.path.join(args.root, CODE_DIR, 'exp_config.json')

with open(EXP_CONFIG_FILE, 'r') as fp:
    params = json.load(fp)[args.exp_config]


params['model_type'] = ModelsType(params['model_type'])

params['seed_value'] = args.seed_value
params['set_seed'] = args.set_seed
params['entity_type'] = args.entity_type
params['dataset'] = args.dataset
params['exp_name'] = args.exp_name
params['exp_config'] = args.exp_config

# All file paths have to be absolute paths
params['WORKING_DIR'] = os.path.join(
    args.root, args.exp_name)
params['DATA_PATH'] = os.path.join(args.data, "NER_data/")
params['CACHE_DIR'] = os.path.join(
    params['WORKING_DIR'], "NER_out_test")

# Where model checkpoints are stored.
params['OUTPUT_DIR'] = os.path.join(
    params['WORKING_DIR'], "model_output_test/")
params['TRAIN_ARGS_FILE'] = os.path.join(
    params['WORKING_DIR'], "train_args_test.json")

if os.path.exists(params['WORKING_DIR']):
    shutil.rmtree(params['WORKING_DIR'])
try:
    os.makedirs(params['WORKING_DIR'])
except OSError as e:
    print(
        f'Dir - {params["WORKING_DIR"]} - {e.os.strerror} . It mean same directory being used for multiple exp')
print(params)
main(params)
