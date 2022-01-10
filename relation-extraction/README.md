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

**All the example code snippets this tutorial are to be run from this folder.**

## (Optional) Debugging with VSCode

First, make sure use your VSCode's "Files -> Open Folders..." options to open `relation-extraction` folder (Note: not the root folder of this project, it's `sbks-ucsd/relation-extraction`). To set up vscode debugging with argument, [create run configurations](https://code.visualstudio.com/docs/python/debugging). You see the existing run configuration in `.vscode/launch.json`

# Data processing pipeline

This section is about everything in `biobert_RE/dataset_processing`. The two goals of the data processing pipeline are:

1. Merging ChemProt and DrugProt datasets
2. Convert the raw data into inputs to relation extraction models

A diagram of the pipeline:

![pipeline diagram](assets/re_pipeline.png)

## article_overlap.py

This script counts the number of articles in each datasets, the number of articles only appear in one dataset (differences), and the number of articles appear in both dataset (intersection). It is used for analyzing the datasets and does not transform/process the data. The usage is as the following:

```
article_overlap.py abs_filename1 abs_filename2
```

* `abs_filename1`: path to the first abstract tsv file
* `abs_filename2`: path to the second abstract tsv file

(Example) Count the number of unique/common articles between ChemProt train dataset and DrugProt  train dataset:

```
python biobert_RE/dataset_processing/article_overlap.py data/ChemProt/training/chemprot_training_abstracts.tsv data/DrugProt/training/drugprot_training_abstracts.tsv
```

### Note

1. article_ids are used to compute overlap since they are unique even across datasets.

## dataset_to_json.py

Both the ChemProt and DrugProt consists of at least three tsv files. They are abstracts, entities, and relations tsv files. This script reads in the three tsv files and converts them into a single json dataset file while performing serveral sanity checks to make sure the raw tsv files are in good format. Converting the original ChemProt and DrugProt into json datasets significantly simplifies the dataset merging step.

```
dataset_to_json.py ent_file, rel_file, abs_file -o out_file [-v]
```

* `ent_file`: path to entity tsv file
* `rel_file`: path to relation tsv file
* `abs_file`: path to abstract tsv file
* `-o out_file`: path to write the json file output
* `-v`: verbose mode

(Example) Convert ChemProt training to json format:

```
python biobert_RE/dataset_processing/dataset_to_json.py data/ChemProt/training/chemprot_training_entities.tsv data/ChemProt/training/chemprot_training_relations.tsv data/ChemProt/training/chemprot_training_abstracts.tsv -o data/ChemProt/training/chemprot_training.json
```

The content of the json file are as the following:

```
{
    <article id (string)>: {
        "abstracts": [
            {
                "text": <a sentence (string)>,
                "start": <start position of the sentence in characters (int)>,
                "end": <end position of the sentence in characters, exclusive (int)>,
                "entities": [
                    {
                        "id": <article-unique id of this entity (string)>,
                        "text": <actual text of this entity (string)>,
                        "type": <type of this entity (string)>,
                        "start": <start position of this entity in characters (int)>,
                        "end": <end position of this entity in characters (int)>
                    }, ...
                ],
                "relations": [
                    {
                        "rel_type": <type of this relation (string)>,
                        "ent_id1": <id of the first entity in this relation (string)>,
                        "ent_id2": <id of the second entity in this relation (string)>,
                        "start": <start position of this relation in characters (int)>,
                        "end": <end position of this relation in characters (int)>
                    }, ...
                ]
            }, ...
        ]
    }, ...
}
```

### Note

1. The first item of the array abstracts is always the title of the article. We put it like this because the spans are relative to the begining of the title.
1. All starts and ends are global position relative the first character in the title.
2. For relations, the entity with ent_id1 isn't guarenteed to precede the entity with ent_id2
3. However, both the entities and relations array are sorted according to their starting position.
3. The start and end (span) of a relation if defined by `min(start of ent_id1, start of ent_id2)`, `max(start of ent_id1, start of ent_id2)` respectively. This is used to check relations that span across multiple sentences (This should not happen, but as discussed on 2021-10-28 in the [meeting note](https://docs.google.com/document/d/15pfeEnx7NxEfTdIYVHPnDhQzBGS7rff-eY1datWWuR4/edit), sometimes SciSpacy makes mistakes during sentence splitting. When this happens, relations spanning multiple sentences are discarded.)

## compare_datasets.py

This script goes through two json datasets A and B, and if it finds an article that exists in both dataset A and dataset B, it will compare their texts, entities, and relations and print out any differences. By supplying an optional path argument `brat_diff_dir`, the script will output the entity and relation differences in brat format (.txt and .ann) under the supplied path.

```
compare_datasets.py [-d brat_diff_dir] json_filename1 json_filename2
```

* `json_filename1`: path to the first json dataset
* `json_filename2`: path to the second json dataset
* `-d brat_diff_dir`: path to write the article differences in brat format

(Example) Find all the entity and relation differences for all articles that exists in both ChemProt  train and DrugProt train datasets, and output the result in brat format to `cp_vs_dp/train`:

```
python biobert_RE/dataset_processing/compare_datasets.py data/ChemProt/training/chemprot_train.json data/DrugProt/training/drugprot_train.json -d cp_vs_dp/train
```

## merge_json_datasets.py

Merge one json dataset with others. Articles that are unique across datasets are added without modification, and articles that exists in multiple datasets are added by merging their relations together. **This script only merges relations so it assumes the texts and entities are identical** (For ChemProt and DrugProt this is the case, but for future datasets cautions are needed to make sure the assumption still holds). During the merging process, specific relation classes are mapped to CPR-X classes.

![cpr_mapping.png](assets/cpr_mapping.png)

The usage of the script is as the following:

```
merge_json_datasets.py --datasets dataset1 [dataset2 ...] --output output_path
```

* `--datasets dataset1 [dataset2 ...]`: a list of json dataset (dataset1, dataset2, ...) to be merged together
* `--output output_path`: the path to write the resulting merged dataset

(Example) Merging ChemProt and DrugProt training together:

```
python biobert_RE/dataset_processing/merge_json_dataset.py --datasets data/ChemProt/training/chemprot_train.json data/DrugProt/training/drugprot_train.json --output data/merged/training/merged.json
```

### Note
1. [Meeting note on 2021-12-16](https://docs.google.com/document/d/15pfeEnx7NxEfTdIYVHPnDhQzBGS7rff-eY1datWWuR4/edit): There are some disagreement between the ChemProt and DrugProt, since the number of disagreement is small, they are discarded.

## json_to_input.py

This script converts json dataset to inputs that can be passed into RE models. It go though all the chemical-gene relations and generate an input sentence for each. The chemical and gene in the input sentence are masked by generic tokens `@CHEMICAL#` and `@GENE#` respectively. To generate negative samples (the NOT relation), It generate input sentences with the NOT label for every chem-gene entity pair that is not identified to have a relation between them.

```
json_to_input.py [-i target_id] json_filename out_filename
```

* `json_filename`: path to the json dataset file
* `out_filename`: path to write the output
* `-i target_id`: if given, will only process the article whose article_id == target_id (useful for debug)

(Example) Turn the merged json dataset to inputs

```
python biobert_RE/dataset_processing/merge_json_dataset.py data/merged/training/merged.json data/merged/training/merged.txt
```

The content of the output is a tsv file with the following columns in order:

1. input_id
2. label
3. input (masked) sentence
4. original sentence

### Note

1. The format of input_id of positive sample is: `{article id}_{sentence index}_{relation index}`. For example, an input_id of 123_0_2 means this sentence is generated from the third relation in the first sentence of the article whose article_id is 123.
2. The format of input_id of negative sample is: `{article id}_{sentence index}_{entity index 1}_{entity index 2}`. For example, an input id of 321_1_0_1 corresponds to a negative sentence generated by masking the first and the second entities in the second sentence of the article whose article_id is 321.