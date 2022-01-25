# sbks_topic_modeling

## Running the Project
* After cloning the program, you need to first download `mallet-2.0.8` in order to run Mallet-LDA model.
* To get the data_words.pickle file, run `python run.py create_data_words`
* To get the `pub_year.json` file, run `python run.py pub_year`
* To get the Mallet-Lda/DTM model, run `python run.py model`
* To get the topic proportions output saved in data/json-files, run `python run.py save_to_data`

## Running it on Nautilus
* Start a pod using the topic-modeling.yaml
* Run the project as mentioned above
