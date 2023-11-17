# Getting started

This folder contains all the necessary scripts to perform relation extraction on DrugProt and ChemProt

* `assets`: contains figures and diagrams for this README
* `biobert_RE/dataset_processing`: contains dataset preprocessing codes that convert raw DrugProt and ChemProt data into training/testing datasets
* `biobert_RE/acs_data_processing`: containts the data proprocessing codes for ACS inference
* `biobert_RE/models`: dataloader, RE model definition, and training logics
* `biobert_RE/utils`: frequently used utility function
* `biobert_RE_chemprot`: legacy code for ChemProt datasets, functionalities are already integrated into the new module `biobert_RE`
* `docker`: contains DockerFile used to build containers to run RE models on Nautilus, and yaml files for batch training job on Nautilus

## Environment setup

Run the following command to setup via Anaconda. For PyTorch installation, make sure to pick the correct cudatoolkit version for your GPU.

```
conda create -n torch python=3.8
conda activate torch
conda install -y scipy=1.6.1
conda install -y -c conda-forge notebook scikit-learn tqdm matplotlib ipywidgets
conda install -y -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
conda install -y -c huggingface transformers
pip install wandb (if needed)
```

Run this to install SciSpacy models. If the second command failed to install en_core_sci_sm due to SciSpacy version constraints, check [here](https://allenai.github.io/scispacy/) for the latest en_core_sci_sm model.

```
pip install scispacy==0.4.0
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
```

## Setting path variable

Make sure the append the path to folder (relation-extraction/biobert_RE) to `PYTHONPATH`. You will get local package not found errors if it is not set up correctly.

## Data source

[ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)

[DrugProt](https://zenodo.org/record/5119892#.YdyOd_7MIUE)

Merged (created by merging ChemProt and DrugProt)

Recommended: after downloading the zip files, unzip them and place them under `relation-extraction/data`. This tutorial will assume the following file structure:

```
relation-extraction
    + data
        + ChemProt
        + DrugProt
        + Merged
```

Check to make sure all datasets have development set and traing set.

ChemProt has also been processed by [Sun et. al.](https://arxiv.org/abs/1911.09487). We are currently not using this processed version and choose to process ChemProt data by ourselves because Sun's version only has CPR 3,4,5,6 and 9.

We are currently using the merged dataset to train our model:
```
merged
    + merged
        + training_original
        + training
            + train.txt
            + vali.txt
        + dev
```
**Note:** Because the dataset don't contain the test set, we will instead use the development set as our test set. The **training-original** contains the original training data; **dev** contains the original development set. We used **train_vali_split.py** to split our training data into 80/20 split to create the **training** which contains the training data and validation data.

## Pretrained weights

[BioBert](https://github.com/dmis-lab/biobert)

Use `transformers-cli` binary come with the `transformers` package to convert Tensorflow weights to PyTorch weights

**All the example code snippets this tutorial are to be run from this folder.**

## (Optional) Debugging with VSCode

First, make sure use your VSCode's "Files -> Open Folders..." options to open `relation-extraction` folder (Note: not the root folder of this project, it's `sbks-ucsd/relation-extraction`). To set up vscode debugging with argument, [create run configurations](https://code.visualstudio.com/docs/python/debugging). You see the existing run configuration in `.vscode/launch.json`

# Data Processing Pipeline

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

## input_to_annbrat.py

This script converts model input format files to .ann files for BRATEval. BRATEval requires two folders: predictions and ground-truth. The script goes through the input file line by line and use the json file to locate back to each entity and the relations. The output folder will contain all .ann files for each article and its respective entities and relations. (It is also used for converting model's output.tsv to the annbrat formats.)

```
input_to_annbrat.py --json_filename JSON_PATH --input_file INPUT_PATH --ann_folder FOLDER_PATH
```

* `JSON_PATH`: path to the json dataset file
* `INPUT_PATH`: path to input.txt file
* `FOLDER`: the output folder for .ann files

(Example) Generate the test ground-truth folder

```
python biobert_RE/dataset_processing/input_to_annbrat.py data/merged/dev/merged.json data/merged/dev/merged.txt data/merged/dev_gold
```

The content of the output are a list of .ann files:

```
vali_gold
    + 00001.ann
    + 00002.ann
    + 00003.ann
    ...
```


# Relation Extraction Model
This section is about everything in `biobert_RE/models`. The module diagram is as the following:

![re_diagram.png](assets/re_diagram.png)

## train.py

This file contains the main training logic of the RE model. It accepts many optional parameters to help configurate the experiment. 

```
train.py 
        [--epochs EPOCHS] 
        [--valid-freq VALID_FREQ] 
        [--warm-up WARM_UP] 
        [--lr LR]
        [--batch-size BATCH_SIZE] 
        [--valid-batch-size VALID_BATCH_SIZE] 
        [--ga-batch-size GA_BATCH_SIZE]
        [--dataloader-workers DATALOADER_WORKERS] 
        [--seq-len SEQ_LEN] 
        [--bert-state-path BERT_STATE_PATH]
        [--use-bert-large USE_BERT_LARGE] 
        [--top-hidden-size TOP_HIDDEN_SIZE [TOP_HIDDEN_SIZE ...]]
        [--freeze-bert FREEZE_BERT] 
        [--resume-from-ckpt RESUME_FROM_CKPT]
        [--resume-from-step RESUME_FROM_STEP] 
        [--train-data TRAIN_DATA]
        [--valid-data VALI_DATA]
        [--test-data TEST_DATA]
        [--inference-data INFERENCE_DATA]
        [--ckpt-dir CKPT_DIR] 
        [--balance-dataset BALANCE_DATASET] 
        [--do-train DO_TRAIN]
        [--do-inference DO_INFERENCE]
        [--record-activation ACTIVATION_LAYERS]
        [--record-wandb PROJECT_NAME]
```

* `epochs`: total number of training epochs
    * Regardless of `valid-freq`, a validation run will be performed after the final epoch.
* `valid-freq`: number of steps between every validation run
    * For example, if `valid-freq` is set to 1000, after 1000, 2000, ... steps, this script will perform a validation on the model. One step is one call to the `step()` method of the optimizer
* `warm-up`: number of steps in warm up stage
    * `lr = lr_factor * min(step ** (-0.5), step * warm_up ** (-1.5))`
    * Adjust `lr_factor` using the `lr` option.
    * This option is ignore if `freeze-bert` is True
* `lr`: learning rate/variable lr factor (see note)
    * If `freeze-bert` is False, this will set the `lr_factor` above.
    * If `freeze-bert` is True, this will set the learning rate of the optimizer.
* `batch-size`: number of samples per forward pass during training
* `valid-batch-size`: number of samples per forward pass during validation
    * We usually can fit more sample per batch during validation because it doesn't cost as much memory.
    * If not specified, it will be set to `batch-size`
* `ga-batch-size`: (gradient accumulation) minimum number of samples the model sees before an update step
    * There is a counter in the script counting how many samples the model has seen since the last step. If the count is greater or equal to `ga-batch-size`, the model will take a step, and the counter will be zeroed.
    * If not specified, it will be set to `batch-size`
* `dataloader-workers`: number of CPU threads dedicated to data loading
* `seq-len`: maximum input sentence length
    * Anything longer than `seq-len` is truncated.
* `bert-state-path`: path to the pretrained bert directory
* `use-bert-large`: whether to use bert large instead of bert base
* `top-hidden-size`: array of integer specifying the size of each hidden layer
    * the length of the array determines the number of hidden layers
* `freeze-bert`: whether to freeze the weights of the entire bert encoder
    * Achieved by disabling the gradient of bert.
* `resume-from-ckpt`: path to the saved checkpoint
* `resume-from-step`: number of steps the saved checkpoint has made
    * Doesn't matter if you are resuming for top model training.
    * **Does matter** if you are resuming for end-to-end training, this tells the optimizer to calculate the learning rate correctly.
* `train-data`: path to the input data
    * ignored when `do-train` is false
* `valid-data`: path to the validation data
    * if not specified, it will be a random split of the training and validation (80/20 split).
* `test-data`: path to the data for the test data set
* `inference-data`: path to the data for inference task (e.g. ACS data)
    * ignored when `do-inference` is false
* `balance-dataset`: whether or not to use stratified sampling during training
* `do-train`: train the model on train data
* `do-inference`: make prediction on the inference data with the model
    * by setting `do-train` and `do-inference` to True, you can train and predict in one go
    * if you want to do prediciton using a specific checkpoint, set `do-train` to False and make sure to set the `resume-from-ckpt` parameter
* `do-brateval`: This will perform prep code for doing BratEval. It will make predictions using the model and write results to `re_output.tsv` and then using that to create .ann files in the `eval` folder.
* `ckpt-dir`: path to the folder that stores model as checkpoints
* `activation`: activation method for the top model. Options include `['ReLU', 'Tanh', 'GELU']`.
* `record-activation`: a list of layers' activations to be recorded and plotted.
    * input type: integer(s) which denotes the layer index(indices)
* `record-wandb`: the name of the wandb project, default to be an empty string `''` which disables wandb recording.

### Example output

The validation output looks like the following:

```
---Results---
Total tested: 1600.0
ACC: 0.711875
Precision: [nan, nan, nan, 0.38652482269503546, nan, nan, nan, 0.7814871016691958]
Recall: [0.0, 0.0, 0.0, 0.6193181818181818, 0.0, 0.0, 0.0, 0.9221128021486124]
F1 Scores [nan, nan, nan, 0.4759825327510917, nan, nan, nan, 0.8459958932238193]
Confusion Matrix:
[[   0.    0.    0.    0.    0.    0.    0.   24.]
 [   0.    0.    0.   13.    0.    0.    0.   61.]
 [   0.    0.    0.   27.    0.    0.    0.   38.]
 [   0.    0.    0.  109.    0.    0.    0.   67.]
 [   0.    0.    0.    7.    0.    0.    0.    8.]
 [   0.    0.    0.   27.    0.    0.    0.   17.]
 [   0.    0.    0.   12.    0.    0.    0.   73.]
 [   0.    0.    0.   87.    0.    0.    0. 1030.]]
```

Precision, recall, and F1 are k element vectors, where the i-th element is the class-wise precision, recall and F1 scores, respectively. The columns of confusion matrix are predictions and rows are ground truth. The example output above is captured from a very early stages of training. The confusion matrix indicates the model is classifying most of the samples as class 3 and 7.

## dataloader.py

Dataloader loads the output of json_to_input.py into memory, uses wordpiece tokenizer to tokenize them into token ids, and create mini batches for training the model.

## nets.py

Model diagram:

![re_model.png](assets/re_model.png)

### Top model

Top model consists of N hidden layers and 1 output layer. The activation function is Tanh, and there is a dropout layer before every layers. The width and depth of the top model is configurable via the training script.

## optimizer.py

Implement variable learning for the bert model. This is enabled only during end-to-end training. The learning rate is calculated as

```
lr = lr_factor * min(step ** (-0.5), step * warm_up ** (-1.5))
```

The following is the learning rate curve with lr_factor=0.0005, warm_up=1000. During the warm up stage, the learning rate increases linearly. Once the warm up finishes, the learning rate is equal to the inversed squared-root of step count.

![var_lr.png](assets/var_lr.png)

# BratEval

[BratEval](https://github.com/READ-BioMed/brateval) is a tool that performs pairwise comparison of annotation sets done on the same set of documents. 

## Requirement
- BratEval modules from its github page
- maven
- java JDK

## How-To on a local machine
- File Format:
```
+ data
    + merged
        + brat_eval
            + brateval-toolkit
            + dev_gold
            + dev_pred
            + vali_gold
            + vali_pred
            re_TEST_output.tsv (produced as output after your run do-brateval in the training script)
```

- Required Format:
    - two folders: gold (ground truth) & eval (predictions)

For example, if we want to evaluate using the validation dataset: we would have two folders in the following format:
```
vali_gold
    + 00001.ann
    + 00002.ann
    + 00003.ann
    ...

vali_eval
    + 00001.ann
    + 00002.ann
    + 00003.ann
    ...
```
**Note:** To generate the gold folder, you will need to run the `input_to_annbrat.py` described above. 

BratEval will try to match each .ann within each folder and perform pairwise comparison with strict match.

- Example of running the BratEval Tool and output:

Command to Run the BratEval on the Test(Dev) set (must be in the folder of BratEval): 
```
mvn exec:java -Dexec.mainClass=au.com.nicta.csp.brateval.CompareRelations -Dexec.args="-e ../dev_pred -g ../dev_gold -s exact"
```
Output:
```
Summary:
CPR-1|CHEMICAL|GENE|tp:166|fp:10|fn:30|precision:0.9432|recall:0.8469|f1:0.8925|fpm:0|fnm:0   
CPR-1|CHEMICAL|GENE-N|tp:38|fp:6|fn:3|precision:0.8636|recall:0.9268|f1:0.8941|fpm:0|fnm:0    
CPR-1|CHEMICAL|GENE-Y|tp:107|fp:11|fn:5|precision:0.9068|recall:0.9554|f1:0.9304|fpm:0|fnm:0  
CPR-1|GENE-N|CHEMICAL|tp:0|fp:3|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-1|GENE-Y|CHEMICAL|tp:0|fp:9|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-1|GENE|CHEMICAL|tp:0|fp:2|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0       
CPR-2|CHEMICAL|GENE|tp:283|fp:235|fn:117|precision:0.5463|recall:0.7075|f1:0.6166|fpm:0|fnm:0 
CPR-2|CHEMICAL|GENE-N|tp:49|fp:9|fn:181|precision:0.8448|recall:0.2130|f1:0.3403|fpm:0|fnm:0  
CPR-2|CHEMICAL|GENE-Y|tp:208|fp:32|fn:340|precision:0.8667|recall:0.3796|f1:0.5279|fpm:0|fnm:0
CPR-2|GENE-N|CHEMICAL|tp:0|fp:2|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-2|GENE-Y|CHEMICAL|tp:0|fp:4|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-2|GENE|CHEMICAL|tp:0|fp:129|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-3|CHEMICAL|GENE|tp:343|fp:36|fn:91|precision:0.9050|recall:0.7903|f1:0.8438|fpm:0|fnm:0   
CPR-3|CHEMICAL|GENE-N|tp:128|fp:8|fn:31|precision:0.9412|recall:0.8050|f1:0.8678|fpm:0|fnm:0  
CPR-3|CHEMICAL|GENE-Y|tp:327|fp:17|fn:64|precision:0.9506|recall:0.8363|f1:0.8898|fpm:0|fnm:0 
CPR-3|GENE-N|CHEMICAL|tp:0|fp:1|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-3|GENE-Y|CHEMICAL|tp:0|fp:4|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0     
CPR-3|GENE|CHEMICAL|tp:0|fp:16|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-4|CHEMICAL|GENE|tp:1023|fp:104|fn:96|precision:0.9077|recall:0.9142|f1:0.9110|fpm:0|fnm:0
CPR-4|CHEMICAL|GENE-N|tp:372|fp:9|fn:38|precision:0.9764|recall:0.9073|f1:0.9406|fpm:0|fnm:0
CPR-4|CHEMICAL|GENE-Y|tp:621|fp:33|fn:67|precision:0.9495|recall:0.9026|f1:0.9255|fpm:0|fnm:0
CPR-4|GENE-N|CHEMICAL|tp:0|fp:10|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-4|GENE-Y|CHEMICAL|tp:0|fp:9|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-4|GENE|CHEMICAL|tp:0|fp:28|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-5|CHEMICAL|GENE|tp:93|fp:2|fn:17|precision:0.9789|recall:0.8455|f1:0.9073|fpm:0|fnm:0
CPR-5|CHEMICAL|GENE-N|tp:21|fp:1|fn:0|precision:0.9545|recall:1.0000|f1:0.9767|fpm:0|fnm:0
CPR-5|CHEMICAL|GENE-Y|tp:95|fp:0|fn:0|precision:1.0000|recall:1.0000|f1:1.0000|fpm:0|fnm:0
CPR-5|GENE-N|CHEMICAL|tp:0|fp:1|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-5|GENE-Y|CHEMICAL|tp:0|fp:8|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-5|GENE|CHEMICAL|tp:0|fp:9|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-6|CHEMICAL|GENE|tp:161|fp:8|fn:9|precision:0.9527|recall:0.9471|f1:0.9499|fpm:0|fnm:0
CPR-6|CHEMICAL|GENE-N|tp:40|fp:2|fn:2|precision:0.9524|recall:0.9524|f1:0.9524|fpm:0|fnm:0
CPR-6|CHEMICAL|GENE-Y|tp:145|fp:9|fn:11|precision:0.9416|recall:0.9295|f1:0.9355|fpm:0|fnm:0
CPR-6|GENE-N|CHEMICAL|tp:0|fp:2|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-6|GENE|CHEMICAL|tp:0|fp:6|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-9|CHEMICAL|GENE|tp:338|fp:16|fn:121|precision:0.9548|recall:0.7364|f1:0.8315|fpm:0|fnm:0
CPR-9|CHEMICAL|GENE-N|tp:105|fp:10|fn:7|precision:0.9130|recall:0.9375|f1:0.9251|fpm:0|fnm:0
CPR-9|CHEMICAL|GENE-Y|tp:275|fp:14|fn:70|precision:0.9516|recall:0.7971|f1:0.8675|fpm:0|fnm:0
CPR-9|GENE-N|CHEMICAL|tp:0|fp:9|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-9|GENE-Y|CHEMICAL|tp:0|fp:64|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
CPR-9|GENE|CHEMICAL|tp:0|fp:39|fn:0|precision:0.0000|recall:0.0000|f1:0.0000|fpm:0|fnm:0
NOT|CHEMICAL|GENE|tp:3196|fp:432|fn:362|precision:0.8809|recall:0.8983|f1:0.8895|fpm:0|fnm:0
NOT|CHEMICAL|GENE-N|tp:1360|fp:246|fn:29|precision:0.8468|recall:0.9791|f1:0.9082|fpm:0|fnm:0
NOT|CHEMICAL|GENE-Y|tp:2359|fp:501|fn:60|precision:0.8248|recall:0.9752|f1:0.8937|fpm:0|fnm:0
NOT|GENE-N|CHEMICAL|tp:1122|fp:0|fn:28|precision:1.0000|recall:0.9757|f1:0.9877|fpm:0|fnm:0
NOT|GENE-Y|CHEMICAL|tp:2198|fp:0|fn:98|precision:1.0000|recall:0.9573|f1:0.9782|fpm:0|fnm:0
NOT|GENE|CHEMICAL|tp:2853|fp:0|fn:229|precision:1.0000|recall:0.9257|f1:0.9614|fpm:0|fnm:0
all|tp:18026|fp:2106|fn:2106|precision:0.8954|recall:0.8954|f1:0.8954|fpm:0|fnm:0
```
**Note: This result should match exactly as our model's code.**

# ACS Data Processing

This module contains all the necessary codes to run inference on ACS data. The overall workflow of the ACS inference task is as follows:

![acs_data_processing.png](assets/acs_data_processing.png)

## acs-re-preprocess.py

This scripts reads ACS data with NER annotation in brat format, converts them into model inputs.

```
acs_re_preproess.py --dataset-dir DATASET_DIR
```

* `dataset-dir`: the directory to the ACS data in brat format

### Note

The ACS data folder has to be in a specific structure for this script to work. Namely, the .txt and .ann files need to be group into separate sub-folders by their article ids. For example:

```
acs-data-folder
    + 00001
        + 00001.txt
        + 00001.ann
    + 00002
        + 00002.txt
        + 00002.ann
    ...
```

Once the script finishes, a `re_input.tsv` will be created under each sub-folder:

```
acs-data-folder
    + 00001
        + 00001.txt
        + 00001.ann
        + re_input.tsv
    + 00002
        + 00002.txt
        + 00002.ann
        + re_input.tsv
    ...
```

You can leave the .tsv files as they are; the `train.py` script will handle the dataloading.

## acs-re-postprocess.py

Once the model finishes inference, a `re_output.tsv` will be created underr each sub-folder:

```
acs-data-folder
    + 00001
        + 00001.txt
        + 00001.ann
        + re_input.tsv
        + re_output.tsv
    + 00002
        + 00002.txt
        + 00002.ann
        + re_input.tsv
        + re_output.tsv
    ...
```

Running this scripts will put all the results in re_output.tsv back into the .ann files. At the end the .ann file should have relations detected by the model.

```
acs_re_postproess.py --dataset-dir DATASET_DIR
```

* `dataset-dir`: the directory to the ACS data.

# Training with Nautilus

## Instructions
### 1. Download the relation-extraction/docker folder to the local system.
### 2. Use `generate_yml_files.py` to generate yml files for running experiments. (see instructions below)
- docker/pods contains the yml files for cpu/gpu pods.
- docker/yml_arguments contains the arguments to be filled in for yml_templates
- docker/yml_templates contains sample gpu job templates with missing arguments
### 3. Be careful with GPU usage policies. (see specifics on Nautilus website)

<br/>

`docker/generate_yml_files.py` is a tool for generating multiple yml manifests using a template file and an argument file. It is useful for running multiple experiments/jobs on Nautilus.

```
generate_yml_files.py 
    --yml-temp YML_TEMP 
    --yml-args YML_ARGS 
    --output OUTPUT
```

* `yml-temp` path to the yml template file
* `yml-args` path to the yml arguement file
* `output` output directory

A yml argument file is a tsv file and each line is defined as the following

```
argument_0    argument_1    argument_2    ...
```

Each line should have the same number of arguments separated by the tab character.

The yml template file should contain special patterns which will be replaced by the actual arguments

```
This is an example template file [[[[ARG:0]]]]:
patterns will be replaced by the actual arguments [[[[ARG:2]]]]
[[[[ARG:1]]]]
```

Running the script using the examples above will yield the following result

```
This is an example template file argument_0:
patterns will be replaced by the actual arguments argument_2
argument_1
```

<br/>

## Some useful commands for Nautilus

-- Creating/Deleting pods:

`kubectl create (apply)/delete -f <yaml_file.yaml>`

-- Start a interactive bash with the pod:

`kubectl exec -it <pod_name> -- /bin/bash`

-- Start jupyter notebook

`jupyter notebook --ip='0.0.0.0' --allow-root`

-- Port forwarding

`kubectl port-forward <pod_name> 8888:8888`

For more commands, refer to [Nautilus Setup](https://docs.google.com/document/d/1WRi9hVpUuFzOkLeF7fkk55jFCRSOa_i3wkTivctt_Os/edit).

For help, refer to [Nautilus Documentation](https://docs.nationalresearchplatform.org/).

<br/>

# Attention Visualization
`biobert_RE/attention_visualization.ipynb` contains the neccessary script to visualize the trained attention weights. The script allows for sampled sentences to be tokenized and visualized using the trained Bert model.

## Requirements:
### System:
```
torch
transformers
bertviz
```
### User-provided:
```
BertLarge (weights/biobert_large_v1.1_pubmed_torch) # can use base model as well
trained model (weights/experiment/checkpoint...) # checkpoint of any trained BioBert model
```

## Example

![attention_viz.png](assets/attention_viz.png)

The above example shows the how the sample sentence "Hello World" is visualized using the Bertviz tool. User can choose the layer number and which attention head to visualize. The mouse hover on [CLS] shows the attentions for this hovered on token. The strength of the attention is seen through the weight of the color.

