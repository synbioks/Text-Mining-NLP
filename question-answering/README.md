# BioLinkBERT
This repo contains a few experiments run on the BioLinkBERT code from https://github.com/michiyasunaga/LinkBERT. LinkBERT model is explained [here] (https://arxiv.org/abs/2203.15827) (ACL 2022). Our use case is focused on Question Answering in the biomedical domain and as such, we have only tested the BioASQ and PubmedQA datasets.

## 1. Set up environment and data
### Environment
Run the following commands to create a conda environment:
```bash
conda create -n linkbert python=3.8
source activate linkbert
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.9.1 datasets==1.11.0 fairscale==0.4.0 wandb sklearn seqeval
```

This src folder contains all the necessary scripts to perform QA tasks

run_examples_blurb_biolinkbert-base.sh: contains scripts to run all the blurb tasks. We comment out everything except bioASQ and PubmedQA

src/seqcls: contains trainer and the main function which is called by our script

docker: contains yaml file to create a pod on Nautilus where we can run the code in the following steps:

### Data
You can download the preprocessed datasets on which we evaluated LinkBERT from [**[here]**](https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip). Simply download this zip file and unzip it.
This includes:
- [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) question answering datasets (HotpotQA, TriviaQA, NaturalQuestions, SearchQA, NewsQA, SQuAD)
- [BLURB](https://microsoft.github.io/BLURB/) biomedical NLP datasets (PubMedQA, BioASQ, HoC, Chemprot, PICO, etc.)
- [MedQA-USMLE](https://github.com/jind11/MedQA) biomedical reasoning dataset.
- [MMLU-professional medicine](https://github.com/hendrycks/test) reasoning dataset.

They are all preprocessed in the [HuggingFace dataset](https://github.com/huggingface/datasets) format.

If you would like to preprocess the raw data from scratch, you can take the following steps:
- First download the raw datasets from the original sources by following instructions in `scripts/download_raw_data.sh`
- Then run the preprocessing scripts `scripts/preprocess_{mrqa,blurb,medqa,mmlu}.py`.

### Run scripts in a nautilus pod
1. clone this git repository
2. cd linkbert
3. wget https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip
4. unzip data
This code automatically logs to wandb as we have wandb installed. Set report_to=none in the run_examples_blurb_biolinkbert-base.sh file to prevent logging to wandb.
5. wandb login (optional) and Enter the API key when prompted
6. chmod +x run_examples_blurb_biolinkbert-base.sh
7. ./run_examples_blurb_biolinkbert-base.sh


## Citation
If you find our work helpful, please cite the following:
```bib
@InProceedings{yasunaga2022linkbert,
  author =  {Michihiro Yasunaga and Jure Leskovec and Percy Liang},
  title =   {LinkBERT: Pretraining Language Models with Document Links},
  year =    {2022},  
  booktitle = {Association for Computational Linguistics (ACL)},  
}
```
