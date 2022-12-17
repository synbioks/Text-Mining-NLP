FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt update -y
RUN apt install -y git vim

RUN conda install scipy==1.4.1 notebook scikit-learn tqdm matplotlib ipywidgets transformers scispacy==0.4.0 wandb pandas nltk boto3 
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz