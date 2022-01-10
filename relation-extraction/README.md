# Getting started

This folder contains all the necessary scripts to perform relation extraction on DrugProt and ChemProt

## Environment setup

Run the following command to setup via Anaconda. For PyTorch installation, make sure the pick the correct cudatoolkit version for your GPU.

```
conda create -n torch python=3.8
conda install -y scipy=1.6.1
conda install -y -c conda-forge notebook scikit-learn tqdm matplotlib ipywidgets
conda install -y -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
conda install -y -c huggingface transformers
```

Run this to install SciSpacy models. If the second command failed to install en_core_sci_sm due to SciSpacy version constraints, check [here](https://allenai.github.io/scispacy/) for the latest en_core_sci_sm model.

```
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
```

## Data source

[ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)

[DrugProt](https://zenodo.org/record/5119892#.YdyOd_7MIUE)

Recommended: after downloading the zip files, unzip them and place them under `relation-extraction/data`. This tutorial will assume the following file structure:

```
relation-extraction
    + data
        + ChemProt
        + DrugProt
```

Check to make sure both datasets have development set and traing set.

ChemProt has also been processed by [Sun et. al.](https://arxiv.org/abs/1911.09487). We are currently not using this processed version and choose to process ChemProt data by ourselves because Sun's version only has CPR 3,4,5,6 and 9.

## Setting path variable

Make sure the append the path to this folder (relation-extraction) to `PYTHONPATH`. You will get local package not found errors if it is not set up correctly.

## (Optional) Debugging with VSCode

First, make sure use your VSCode's "Files -> Open Folders..." options to open `relation-extraction` folder (Note: not the root folder of this project, it's `sbks-ucsd/relation-extraction`). To set up vscode debugging with argument, [create run configurations](https://code.visualstudio.com/docs/python/debugging). You see the existing run configuration in `.vscode/launch.json`

# Data processing pipeline

This section is about everything in `biobert_RE/dataset_processing`. The two goals of the data processing pipeline are:

1. Merging ChemProt and DrugProt datasets
2. Convert the raw data into inputs to relation extraction models

A diagram of the pipeline:



command copy-paste board:

process drugprot dev
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./drugprot_development_entities.tsv ./drugprot_development_relations.tsv ./drugprot_development_abstracts.tsv -o ./drugprot_dev.json -v
```

process drugprot train
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./drugprot_training_entities.tsv ./drugprot_training_relations.tsv ./drugprot_training_abstracts.tsv -o ./drugprot_train.json -v
```

process chemprot dev
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./chemprot_development_entities.tsv ./chemprot_development_relations.tsv ./chemprot_development_abstracts.tsv -o ./chemprot_dev.json -v
```

process chemprot train
```
python ../../../biobert_RE_v2/dataset_processing/dataset_to_json.py ./chemprot_training_entities.tsv ./chemprot_training_relations.tsv ./chemprot_training_abstracts.tsv -o ./chemprot_train.json -v
```

compare chemprot and drugprot train
```
python compare_datasets.py ../../data/DrugProt/training/drugprot_train.json ../../data/ChemProt/chemprot_training/chemprot_train.json
```

generate input sentences from json drugprot
```
python json_to_input.py ../../data/DrugProt/training/drugprot_train.json ../../data/DrugProt/training/drugprot_train.txt
```

generate input sentences from json chemprot
```
python json_to_input.py ../../data/ChemProt/chemprot_training/chemprot_train.json ../../data/ChemProt/chemprot_training/chemprot_train.txt
```

merge chemprot with drugprot
```
python biobert_RE_v2/dataset_processing/merge_json_datasets.py --datasets data/DrugProt/training/drugprot_train.json data/ChemProt/chemprot_training/chemprot_train.json --output data/merged/training/merged.json
```

```
python biobert_RE_v2/dataset_processing/merge_json_datasets.py --datasets data/ChemProt/chemprot_training/chemprot_train.json data/DrugProt/training/drugprot_train.json --output data/merged/training/merged.json
```

Some notes:

1. article ids are unique even across datasets
2. drugprot train and dev do not share any common article ids (we assume this is the case for other datasets as well)
3. the abstract in .json datasets are actually abstract + title (the spans start from title)
4. entities and relations in abstract sentences are sorted by start position
5. chemprot and drugprot has the same exact abstracts, and entities