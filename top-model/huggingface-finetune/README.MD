Code Structure:

- run_test.sh
    - Input params
    - <exp_name_to_save>{e.g. baseline_v10_lr5_6_wd3_ft5_pt5_tp0_0909_3} <entity>[Gene] <dataset>[gpro] <exp_config>[baseline_v10_lr5_6_wd3_ft5_pt5_tp0 5]
    - It calls main.py
- Main.py
    - Input args
    --seed_value 42 --entity_type $2 --dataset $3 --data "/sbksvol/nikhil/" --exp_name "$2-$3-$var-$1-$st" --exp_config $4 --root "/sbksvol/nikhil/"
    - Here $num represents a variable from run_test.sh file
    - Update os.environ["WANDB_API_KEY"] to set wandb
    - This file reads all args and config from exp_config and call main code from run_v2.py
    - Configuration is read from exp_config.json
- Run_v2.py
    - Main functions 
        - main(_params): prepare data, train model, and test model in sequence
        - prepare_data(): read and create dataframe and data related info
        - prepare_config_and_tokenizer(): create data and model related configs
        - NerDataset(): pytorch extended dataset implemented in utils_ner.py
        - run_train(): start training and do evaluation on validation data.
        - run_test(): do evaluation on specified model and data
        - It uses models_factory.py to create a model according to the exp configuration. 
- models.py/models_fcn.py:
    - Models.py create CRF models
    - Models_fcn.py creates models with Softmax
- Exp_config.py: description in [square brackets]
```
"fcn_crf_v10_lr6_7_wd5_wdf5_ft5_pt10_tp0": {
        "LOWER_CASE": false, [Cased or Uncased model]
        "LOAD_BEST_MODEL": true,[Load best model for evaluation]
        "MAX_LEN": 256,[Max len of input to BERT]
        "BATCH_SIZE": 32,
        "EPOCH_TOP": 2,[Number of epochs for training just top model]
        "EPOCH_END2END": 30,[Number of epochs for complete model]
        "model_type": 4,[Use models_enum.py for specifying model type]
        "grad_finetune": true,[If finetune complete model for 2 stage]
        "grad_e2e": false,[If run only e2e]
        "grad_finetune_layers": 5,[Number of layers to train in BERT]
        "lr": 5e-6,[lr for top model]
        "weight_decay": 5e-5,[for top model]
        "wd_finetune": 5e-5,[weight_decay for complete model]
        "lr_finetune":5e-7, [lr for complete model]
        "data_size": 100, [% of data to train on]
        "patience": 10, 
        "model_name_or_path":"/sbksvol/nikhil/model/biobert_v1.0_pubmed_pmc/",
        "tf": false, [needed for pytorch]
        "top_model": 0, [0-no hidden, 1- 1 hidden, 3- 3 hidden layers]
        "xargs": 
        {
            "hmask": true,
            "skip_subset": true, [if skip subset except first for gradient]
            "beta1_finetune": 0.95 [beta1 in optimizer]
        }
    }
```
- Bert_job.yaml 
    - This is executed using kubectl to create a batch job which runs the run_test.sh file as a command.
    - Models files are created under sbksvol/<user_dir>



- Models
    - Baseline
        - This consist of BERT + a linear layer to transform 768 to 3 + Softmax
    - Softmax
        - This consist of BERT + 4 linear layers[500,250,125,3] + Softmax
    - CRF
        - This consist of BERT + 4 linear layers[500,250,125,3] + CRF

#### How to run ?
Create a batch file using bert_job.yaml from your machine like 
```
kubectl create -f bert_job.yaml
```
