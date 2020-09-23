import pandas as pd
import numpy as np
import os

# Example: Huner Chemical datasets 
data_dir = "Huner_chemical_data"
datasets = [x for x in os.listdir(data_dir)]

# Suffix of the .CONLL file. 
splits = ["train", "dev", "test"]

for dataset in datasets:

    # The path will be updated soon. 
    root_path = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/"+dataset

    for split in splits:

        file_path = os.path.join(data_dir, dataset, dataset+".conll."+split)

        df = pd.read_csv(file_path, sep='\t', skip_blank_lines=False)
        df = df.fillna("EMPTY_ROW#")

        # Seperate the tokens and tags. 
        tokens = df_dev.apply(lambda x: x.str.split()[0][0], axis=1)
        tags = df_dev.apply(lambda x: x.str.split()[0][-1], axis=1)

        df_out = pd.concat([tokens, tags], axis=1)

        df_out.columns = ['Token', 'Tag']
        df_out['Token'] = df_out['Token'].replace(['EMPTY_ROW#'], '')
        df_out['Tag'] = df_out['Tag'].replace(['EMPTY_ROW#'], '')

        # Replace B-NP with B, Replace I-NP with I. 
        df_out['Tag'] = df_out['Tag'].replace(['B-NP'], 'B')
        df_out['Tag'] = df_out['Tag'].replace(['I-NP'], 'I')

        df_out.reset_index(drop=True)

        df_out.to_csv(os.path.join(root_path, split + ".tsv"), index=False, sep='\t')
