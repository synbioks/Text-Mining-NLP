# sbks_topic_modeling

## Getting started

This folder contains all the necessary scripts to perform topic modeling on ACS and Pubmed data

* `Ethics-ACS`: contains files neccessary for exploring ACS and Pubmed ethics relations
* `Ethics-ACS/Ethics-ACS`: contains output files for two cluster comparisons
* `Ethics-ACS/topic modeing for ACS-Pubmed`: containts scripts to perform topic modeling on both sets of data
* `Ethics-ACS/(notebooks)*.ipynb`: Experiments previously done on the two clusters, including finding divergence between the two topics; PMC_data_extracter includes information for text mining process
* `topic-modeling`: The general script to perform topic modeling on ACS data.
* `previous_code`: legacy code for preprocessing on ACS data and NER data
* `topic-modeling.yaml`: .yaml file for topic-modeling docker container.

## Requirements

```
pip install numpy 
            pandas
            nltk
nltk.download('stopwords')
nltk.download('wordnet')
pip install scikit-learn
            matplotlib
            seaborn
            gensim
Mallet-2.0.8 (May also need to install the latest JVM)
```

## Running the Project
* After cloning the program, you need to first download `mallet-2.0.8` in order to run Mallet-LDA model.
* To get the data_words.pickle file, run `python run.py create_data_words`
* To get the `pub_year.json` file, run `python run.py pub_year`
* To get the Mallet-Lda/DTM model, run `python run.py model`
* To get the topic proportions output saved in data/json-files, run `python run.py save_to_data`

## Running it on Nautilus
* Start a pod using the topic-modeling.yaml
* Run the project as mentioned above
