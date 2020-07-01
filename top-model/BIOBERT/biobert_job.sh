#!/bin/bash
cd /sbksvol/gaurav
source tfenv/bin/activate
cd BiLSTM-CRF

export CUDA_VISIBLE_DEVICES=0,1
python3 top-model.py
# python3 get_all_embeddings.py

## PUT THIS SCRIPT AT /sbksvol/gaurav/