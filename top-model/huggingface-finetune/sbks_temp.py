# Set a seed value
seed_value= 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

data_path = "/sbksvol/nikhil/NER_data/"

from torchcrf import CRF
import wandb

from run_v2 import prepare_data, params, prepare_config_and_tokenizer
import utils_ner
import importlib
importlib.reload(utils_ner)
from utils_ner import NerDataset, Split
from models_factory import get_model
ENT = "Gene"
DATASET = "deca"
import os
data_dir = os.path.join(data_path, ENT, DATASET)
params["entity_type"] = ENT
params["dataset"] = DATASET

params["LOWER_CASE"] = False
params["LOAD_BEST_MODEL"] = True
params["MAX_LEN"] = 256
params["BATCH_SIZE"] = 32
params["EPOCH_TOP"] = 100
params["EPOCH_END2END"] = 100

params["EXP_NAME"] = "test"
params["WORKING_DIR"] = "sbksvol/nikhil/" + params["EXP_NAME"] + "/"
params["CACHE_DIR"] = params["WORKING_DIR"] + "NER_out_test/"

# Where model checkpoints are stored.
params["OUTPUT_DIR"] = params["WORKING_DIR"] + "model_output_test/"
params["TRAIN_ARGS_FILE"] = params["WORKING_DIR"] + "train_args_test.json"
params["DATA_PATH"] = data_path

train_df, test_df, dev_df, labels, num_labels, label_map, data_dir, wt = prepare_data()

data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(
    data_dir, labels, num_labels, label_map)

## Create Dataset Objects
print(Split.dev)


import pandas as pd
def get_data(ENT, DATASET):
    import os
    data_dir = os.path.join(data_path, ENT, DATASET)
    params["entity_type"] = ENT
    params["dataset"] = DATASET

    params["LOWER_CASE"] = False
    params["LOAD_BEST_MODEL"] = True
    params["MAX_LEN"] = 256
    params["BATCH_SIZE"] = 32
    params["EPOCH_TOP"] = 100
    params["EPOCH_END2END"] = 100

    params["EXP_NAME"] = "test"
    params["WORKING_DIR"] = "sbksvol/nikhil/" + params["EXP_NAME"] + "/"
    params["CACHE_DIR"] = params["WORKING_DIR"] + "NER_out_test/"

    # Where model checkpoints are stored.
    params["OUTPUT_DIR"] = params["WORKING_DIR"] + "model_output_test/"
    params["TRAIN_ARGS_FILE"] = params["WORKING_DIR"] + "train_args_test.json"
    params["DATA_PATH"] = data_path

    train_df, test_df, dev_df, labels, num_labels, label_map, data_dir, wt = prepare_data()

    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(
        data_dir, labels, num_labels, label_map)

    dataset =[]
    data_df = [train_df, dev_df, test_df]
    return dataset,tokenizer, data_df

from models.models_enum import ModelsType
import models_factory
import importlib
importlib.reload(models_factory)
from transformers import (
    BertConfig,
    BertTokenizer
)

# config = BertConfig.from_json_file(
#         "/sbksvol/nikhil/model/bert_pt")
# config = BertConfig.from_json_file(
#         "/sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc/bert_config.json")

m_path = "/sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc/"

tokenizer = BertTokenizer.from_pretrained(
        "/sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc",
        cache_dir=".",
        do_lower_case=params["LOWER_CASE"]
    )

from run_v2 import run_test
from transformers import Trainer
from models.models_enum import ModelsType


def get_model(num_labels,label_map,labels, model_path):
    config = BertConfig.from_pretrained(
        "/sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc/",
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir="."
    )
    xargs={'tf':False,'top_model': 3}
    xargs['wt'] = []

    model = models_factory.get_model(
            model_path="/sbksvol/nikhil/"+model_path,
#             model_path="/sbksvol/nikhil/Cellline_cellfinder_baseline1_v10_lr5_6_wd3_ft5_pt5_0208_1/model_output_test/checkpoint-572/",
            cache_dir=".",
            config=None,
            model_type=ModelsType.BASELINE,xargs=xargs)
    # model = BertNERCRFFCN.from_pretrained(
    #             "/sbksvol/nikhil/gene_cell_fcn_crf_lr5_6_wd3/model_output_test/checkpoint-2613/",
    #             xargs = {}
    #         )
    return model

def run_test_exp(ENT, DATASET,model_path):
    data_dir = os.path.join(data_path, ENT, DATASET)
    params["entity_type"] = ENT
    params["dataset"] = DATASET
    
    train_df, test_df, dev_df, labels, num_labels, label_map, data_dir, wt = prepare_data()
    data_args, model_args, config, tokenizer = prepare_config_and_tokenizer(
        data_dir, labels, num_labels, label_map)
    
    model = get_model(num_labels,label_map,labels,model_path)
    trainer = Trainer(model=model)
    params['model_type']=ModelsType.BASELINE

    test_dataset = NerDataset(
        data_dir=data_args['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args['max_seq_length'],
        overwrite_cache=data_args['overwrite_cache'],  # True
        mode=Split.test, data_size=100
    )
    
    run_test(trainer,model,test_dataset, test_df,label_map)
    from seqeval.scheme import IOB2
    run_test(trainer,model,test_dataset, test_df,label_map,strict={'mode':'strict', 'scheme':IOB2})

print("Gene","variome")
m_path = "Gene-variome-2-baseline_v10_lr5_6_wd3_ft5_pt5_tp3_0909_1/model_output_test/checkpoint-4350"
run_test_exp("Gene","variome",m_path)

print("Gene","jnlpba")
m_path = "Gene-jnlpba-4-baseline_v10_lr5_6_wd3_ft5_pt5_tp3_0909_1/model_output_test/checkpoint-348"
run_test_exp("Gene","jnlpba",m_path)

print("Gene","BC2GM")
m_path = "Gene-BC2GM-1-baseline_v10_lr5_6_wd3_ft5_pt5_tp3_0909_1/model_output_test/checkpoint-392"
run_test_exp("Gene","BC2GM",m_path)

print("Chemicals","scai_chemicals")
m_path = "Chemicals-scai_chemicals-2-baseline_v10_lr5_6_wd3_ft5_pt5_tp3_0909_3-1632372558/model_output_test/checkpoint-1170"
run_test_exp("Chemicals","scai_chemicals",m_path)

print("Cellline","jnlpba")
m_path = "Cellline-jnlpba-3-baseline_v10_lr5_6_wd3_ft5_pt5_tp3_dn50_0909_3-1632990311/model_output_test/checkpoint-1605"
run_test_exp("Cellline","jnlpba",m_path)

print("Species-linneaus")
m_path = "Species-linneaus-5-fcn_crf_v10_lr5_6_wd5_wdf5_ft5_pt5_tp3_skip1_0909_1/model_output_test/checkpoint-397"
run_test_exp("Species","linneaus",m_path)