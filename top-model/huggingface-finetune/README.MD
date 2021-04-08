This code is used to finetune BERT models over HUNER data. It can be found on nautilus at /sbksvol/gaurav/NER_src.
<br>
<br>
**To run directly on nautilus:**
<br>
```
cd /sbksvol/gaurav/NER_src/
/root/anaconda3/bin/python3.7 run_ner.py --seed_value 42 --entity_type Cellline --dataset cellfinder --model dense_layer_crf
```
**To run as a batch job:**
```
kubectl create -f bert_job.yaml
```
<br>
The latest docker image used is nakumgaurav/sbks:dev8. The new tag has huggingface transformers library and the seqeval package installed.

<br>
<br>
The code assumes that "NER_out" and "NER_data" directories exist along with "NER_src". NER_data contains the HUNER data and NER_out is used as a cache by the finetuning code.
<br>
<br>
### Main Files:
<br>
run_ner.py -> main script to run the code.
<br>
models.py -> consists of pytorch models for fully connected layers and a generic BertNERTopModel.
<br>
model_utils.py -> Contains helper functions used by models.py.
<br>
data_utils.py -> Helper functions for data preprocessing/loading.
<br>
<br>
### Jupyter Notebooks:
<br>
The same code can be run using the notebook selective_fine_tune.ipynb.
<br>
fine_tune.ipynb uses the BertForTokenClassification model from Huggingface which contains a simple softmax head over BERT for NER.
<br>
selective_fine_tune_modularized.ipynb -> used for testing purposes. Can be ignored
<br>
<br>
### Other file/folders:
<br>
run_ner.sh -> bash script used to run the code.
<br>
bert_job.yaml -> kubernetes yaml file used to run the code as a batch job. It uses run_ner.sh
<br>
model_output -> directory used by run_ner to store model checkpoints
<br>
training_args.json -> created automatically during training phase
<br>
convert_tf_checkpoint_to_pytorch -> script used to convert models trained in tensorflow to pytorch