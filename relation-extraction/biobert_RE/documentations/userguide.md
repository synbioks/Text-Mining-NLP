# Introduction

Huggingface adaptation (work in progress) of the original BioBERT model for extracting relation from ACS articles.

# Setup

In biobert_RE directory, setup the environment using the following command:
```
conda env create -f environment.yml
```

# Pretrained BioBERT

The pretrained BioBERT model can be obtained from [here](https://github.com/dmis-lab/biobert). You need convert the model from tensorflow format to pytorch format. [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html) can help you to do this, it should be installed during the "setup" step above.

# Datasets

[ChemProt](https://biocreative.bioinformatics.udel.edu/) is a dataset containing chemical-gene relations. This dataset is used for fine tuning the BioBERT model.

# Fine Tuning BioBERT

To fine tune the BioBERT, use the following command:
```
python train.py [
    --init_state                <path, checkpoint to resume from>
    --ckpt_dir                  <dir, for saving checkpoints>
    --do_train                  <bool, whether run train or not>
    --train_step                <int, number of training step>
    --num_test_sample           <int, number of testing sample to test after training, 
                                leave this empty to test all the test samples>
    --train_val_frequency       <int, number of step between each validation cycle>
    --data_dir                  <dir, to the fine tune dataset>
    --pretrained_weights_dir    <dir, to the pretrained BioBERT weights>
]
```

# Predicting Relations Using BioBERT

To run relation extraction on ACS articles, you need to first do some preprocess the BRAT NER results. Place each `.txt` and `.ann` pair in separate sub directory and put the sub directories into a parent directory. The directory structure should look like the following:
```
parent-dir
|-- article-1
|   |-- article-1.txt
|   +-- article-1.ann
|--article-2
|   |-- article-2.txt
|   |-- article-2.ann
...
```
Then, use the following command to preprocess the NER results.
```
python acs-re-preprocess.py --dataset_dir=<dir, to the parent directory of the unprocessed BRAT NER results>
```
The preprocess script should generate a `re_input.tsv` file in every sub directory. After this, you can run BioBERT prediction:
```
python predict.py --dataset_dir=<dir, to the parent directory of the processed BRAT NER results>
```
After running the prediction, every sub directory should have a `re_output.tsv` file. Run postprocess on the outputs:
```
python acs-re-postprocess.py --dataset_dir=<dir, to the parent directory of the BioBERT RE results>
```
The postprocess script will append detected relations to the `.ann` files.