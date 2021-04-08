import numpy as np
np.set_printoptions(suppress=True)
import getopt
import sys
import os
from tqdm import tqdm

from datasets import *
from models import *

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer

CKPT_ROOT = "../../weights"
DATASET_ROOT = "../../datasets"
CHEMPROT_ROOT = os.path.join(DATASET_ROOT, "CHEMPROT")

def train_net(task_name, net, train_dataloader, valid_dataloader, loss_fn, optimizer, max_step, valid_freq, ckpt_dir, init_step):

    # setup training
    print(f"Running train {task_name}")
    net.train()
    train_step_count = 0
    epoch_count = 0

    # train loop
    while True:

        print(f"Running epoch {epoch_count}")
        for i, (x, y) in enumerate(train_dataloader):
            
            # decide to break
            if train_step_count >= max_step:
                print(f"Training completed, {train_step_count} steps in total")
                return

            net.train_step(x, y, loss_fn, optimizer)
            train_step_count += 1

            if train_step_count % valid_freq == 0:
                net.eval()
                test_net("TRAIN", net, train_dataloader)
                test_net("VALIDATION", net, valid_dataloader)
                if ckpt_dir is not None:
                    ckpt_path = os.path.join(ckpt_dir, f"{init_step + train_step_count}")
                    torch.save(net.state_dict(), ckpt_path)
                net.train()
                print(f"Resume epoch {epoch_count}")
        epoch_count += 1


def test_net(task_name, net, test_dataloader):

    # setup testing
    print(f"Running test {task_name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = 4
    confusion_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_dataloader):

            pred = net.predict(x_batch).cpu().numpy()

            y_batch = y_batch.numpy()
            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)

            for p, y in zip(pred, y_batch):
                confusion_mat[p][y] += 1

    print("---Results---")
    print("Total tested:", np.sum(confusion_mat))
    print("ACC:", num_correct / num_tested)
    print(confusion_mat)

def predict_net(task_name, net, dataloader):

    print(f"Running prediction {task_name}")
    net.eval()
    res = []
    with torch.no_grad():
        for _, (id1s, id2s, batch_x) in enumerate(dataloader):
            pred, score = net.predict(batch_x, return_score=True)
            score = score.cpu().tolist()
            pred = pred.cpu().tolist()
            for i in range(len(id1s)):
                res.append((id1s[i], id2s[i], pred[i], score[i][pred[i]]))
    return res

def train_cls_end_to_end():

    init_state = None
    top_model_init_state = None
    # init_state = "../../weights/chemprot-cls-end-to-end/5000"
    # top_model_init_state = "../../weights/chemprot-cls-top-model/10000"

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )
    label_filter = ["CPR:3", "CPR:4", "CPR:9", "false"]
    max_seq_len = 128

    # dataloaders
    train_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    train_data.load(os.path.join(CHEMPROT_ROOT, "train_4cls.tsv"))
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        collate_fn=chemprot_tsv_collate_fn
    )

    valid_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    valid_data.load(os.path.join(CHEMPROT_ROOT, "dev_4cls.tsv"))
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        collate_fn=chemprot_tsv_collate_fn
    )

    # net
    init_step = 0
    net = CLSEndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc")
    if init_state is not None:
        state_filename = init_state.split("/")[-1]
        if "-" in state_filename:
            init_step = int(state_filename.split("-")[-1])
        else:
            init_step = int(state_filename)
        net.load_state_dict(torch.load(init_state))
    if top_model_init_state is not None:
        net.top_model.load_state_dict(torch.load(top_model_init_state))
    net = net.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    # train / test loop
    train_net(
        "TRAIN", 
        net, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn, 
        optimizer, 
        10000,  # max step
        1000,   # vaild frequency
        "../../weights/chemprot-cls-end-to-end", # checkpoint directory
        init_step
    ) 

def acs_predict():

    dataset_dir = "../../datasets/acs-20210331-gold"

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )

    net = CLSEndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc")
    net.load_state_dict(torch.load("../../weights/chemprot-cls-end-to-end/top-first-9000"))
    net = net.cuda()
    net.eval()

    for pub_num in tqdm(os.listdir(dataset_dir)): # find all data folder in the dataset directory
        article_dir = os.path.join(dataset_dir, pub_num)
        assert os.path.isdir(article_dir)
        dataset = ACSDataset(
            data_path=os.path.join(article_dir, "re_input.tsv"),
            tokenizer=tokenizer,
            max_seq_len=128
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            num_workers=8,
            shuffle=False,
            collate_fn=acs_collate_fn
        )
        output = predict_net(pub_num, net, dataloader)
        with open(os.path.join(article_dir, "re_output.tsv"), "w", encoding="utf8") as fout:
            fout.write("id1\tid2\tclass\tconfidence\n")
            for _, (id1, id2, pred, score) in enumerate(output):
                fout.write(f"{id1}\t{id2}\t{pred}\t{score}\n")

def train_cls_top_model():

    init_state = None
    init_state = "../../weights/chemprot-cls-top-model/5000"

    # dataloaders
    train_data = ChemprotH5Dataset()
    train_data.load(os.path.join(CHEMPROT_ROOT, "train.h5"))
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        collate_fn=chemprot_h5_collate_fn
    )

    valid_data = ChemprotH5Dataset()
    valid_data.load(os.path.join(CHEMPROT_ROOT, "dev.h5"))
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        collate_fn=chemprot_h5_collate_fn
    )

    # net
    init_step = 0
    net = CLSTopModel()
    if init_state is not None:
        state_filename = init_state.split("/")[-1]
        if "-" in state_filename:
            init_step = int(state_filename.split("-")[-1])
        else:
            init_step = int(state_filename)
        net.load_state_dict(torch.load(init_state))
    net = net.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    # train / test loop
    train_net(
        "TRAIN", 
        net, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn, 
        optimizer, 
        5000,  # max step
        1000,   # vaild frequency
        "../../weights/chemprot-cls-top-model", # checkpoint directory
        init_step
    )     

if __name__ == "__main__":

    # 5e-5 to 1e-5 early stopping
    # top layer pooling

    acs_predict()
    # train_cls_end_to_end()
    # train_cls_top_model()