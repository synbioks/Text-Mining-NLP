import numpy as np
import os

ENTITY = "Chemicals"
DATASETS = [x for x in os.listdir("/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/"+ENTITY)]
if(".DS_Store" in DATASETS):
    DATASETS.remove(".DS_Store")
if(".ipynb_checkpoints" in DATASETS):
    DATASETS.remove(".ipynb_checkpoints")

EMBD_DIR = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/interim/"+ENTITY
SAVE_PATH = "/sbksvol/xiang/sbks_gitlab/top-model/BIOBERT/NER/data/processed/Chemical_512/"

MAX_LEN = 512
EMBEDDING = 768
N_TAGS = 3
CHUNK_SIZE = 16

def chunks(iterable, chunk_size):
    """
    input: 
        iterable: iterable that can be iterated over. 
        chunk_size: number of sentences of each chunk.
    output: 
        generator
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]
    
    
count = 0       
for dataset in DATASETs:
        
    embd = os.path.join(EMBD_DIR, dataset)

    # load the entire embeddding into RAM
    X_train = np.load(os.path.join(embd, "X_train.npy")) 
    y_train = np.load(os.path.join(embd, "y_train.npy"))

    # cast the generator to a list
    X_train_chunk = list(chunks(X_train, CHUNK_SIZE))
    y_train_chunk = list(chunks(y_train, CHUNK_SIZE))

    for i,j in zip(X_train_chunk, y_train_chunk):
        count += 1
        with open(SAVE_PATH+'Train/'+dataset+'_{}.npy'.format(count), 'wb') as f:
            np.save(f, i)
        with open(SAVE_PATH+'Train_label/'+dataset+'_lab_{}.npy'.format(count), 'wb') as f:
            np.save(f, j)
        