import numpy as np
np.set_printoptions(suppress=True)
import getopt
import sys
import os
from tqdm import tqdm
import time

from datasets import *
from models import *

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertModel, BertTokenizer

CKPT_ROOT = "../../weights"
DATASET_ROOT = "../../datasets"
CHEMPROT_ROOT = os.path.join(DATASET_ROOT, "CHEMPROT")

# variable learning rate
class NoamOptim:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, init_step, warmup, factor, optimizer):
        self.optimizer = optimizer
        self._step = init_step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate2()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def rate2(self, step=None):
        if step is None:
            step = self._step
        return self.factor * min(step ** (-0.5), step * self.warmup ** (-1.5))

def train_net(task_name, net, train_dataloader, valid_dataloader, loss_fn, optimizer, n_epochs, valid_freq, ckpt_dir, init_step):

    # setup training
    print(f"Running train {task_name}")
    net.train()
    train_step_count = 0
    epoch_count = 0

    # train loop
    for epoch_count in range(n_epochs):
        print(f"Running epoch {epoch_count}\n")
        for i, (x, y) in enumerate(train_dataloader):
            print("Running for epoch {} batch {} train step {}".format(epoch_count, i, train_step_count), end = "\r")
            # decide to break
            #if train_step_count >= max_step:
            #    print(f"Training completed, {train_step_count} steps in total")
            #    return

            net.train_step(x, y, loss_fn, optimizer)
            train_step_count += 1

            if train_step_count % valid_freq == 0:
                net.eval()
                print(f"Step {init_step + train_step_count} finished")
                test_net("TRAIN", net, train_dataloader)
                test_net("VALIDATION", net, valid_dataloader)
                if ckpt_dir is not None:
                    a = time.time()
                    ckpt_path = os.path.join(ckpt_dir, f"{init_step + train_step_count}")
                    torch.save(net.state_dict(), ckpt_path)
                    print(time.time() - a)
                net.train()
                print(f"Resume epoch {epoch_count}")

def test_net(task_name, net, test_dataloader):

    # setup testing
    print(f"Running test {task_name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = 4
    confusion_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(enumerate(test_dataloader)):

            pred = net.predict(x_batch).cpu().numpy()

            y_batch = y_batch.numpy()
            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)

            for p, y in zip(pred, y_batch):
                confusion_mat[p][y] += 1

    recalls = []
    precisions = []
    f1s = []
    for i in range(num_classes):
        recall = confusion_mat[i][i] / np.sum(confusion_mat[:, i])
        precision = confusion_mat[i][i] / np.sum(confusion_mat[i])
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(2 * recall * precision / (recall + precision))

    print("---Results---")
    print("Total tested:", np.sum(confusion_mat))
    print("ACC:", num_correct / num_tested)
    print("Precision:", precisions)
    print("Recall:", recalls)
    print("F1 Score", f1s)
    print("F1 three cls", np.mean(f1s[:3]))
    print("Confusion Matrix:")
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

def train_cls_end_to_end(batch_size=32, max_seq_len=128):

    init_state = None
    top_model_init_state = None
    # init_state = "../../weights/chemprot-cls-end-to-end/top-first-9000"
    # top_model_init_state = "../../weights/chemprot-cls-top-model/10000"

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )
    label_filter = ["CPR:3", "CPR:4", "CPR:9", "false"]

    # dataloaders
    train_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    train_data.load(os.path.join(CHEMPROT_ROOT, "train_4cls.tsv"))
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        collate_fn=chemprot_tsv_collate_fn
    )

    valid_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    valid_data.load(os.path.join(CHEMPROT_ROOT, "dev_4cls.tsv"))
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        collate_fn=chemprot_tsv_collate_fn
    )
    
    test_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    test_data.load(os.path.join(CHEMPROT_ROOT, "test_4cls.tsv"))
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        collate_fn=chemprot_tsv_collate_fn
    )


    # net
    init_step = 0
    net = EndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc", top_model=CLSTopModel())
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
    adam = optim.AdamW(net.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    optimizer = NoamOptim(768, init_step, 3000, 0.0005, adam) 
    # current best 768, init_step, 3000, 0.0005, adam, rate2() func
    # current best 768, init_step, 3000, 0.02, adam, rate() func

    print(net)

    print("Dataset statistics : ")
    print("Train / Val / Test : {} / {} / {}".format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))
    print("Training model with params bs/seqlen : {} {}".format(batch_size, max_seq_len))

    # train / test loop
    train_net(
        "TRAIN", 
        net, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn, 
        optimizer, 
        14,  # max step
        1000,   # vaild frequency
        "../../weights/chemprot-cls-end-to-end", # checkpoint directory
        init_step
    ) 

    test_net("TEST", net, test_dataloader)


class activation_hook:

    def __init__(self):
        self.activations = []

    def __call__(self, net, x, y):
        self.activations.append(y.detach().cpu().numpy())

    def get_act(self):
        res = []
        for batch in self.activations:
            for item in batch:
                res.append(item)
        return np.array(res)

def gen_chemprot_act(model_name):

    init_state = f"../../weights/chemprot-cls-end-to-end/{model_name}"
    
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )
    label_filter = ["CPR:3", "CPR:4", "CPR:9", "false"]
    max_seq_len = 128

    test_data = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    test_data.load(os.path.join(CHEMPROT_ROOT, "test_4cls.tsv"))
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        collate_fn=chemprot_tsv_collate_fn
    )

    net = EndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc", top_model=CLSTopModel())
    net.load_state_dict(torch.load(init_state))
    net = net.cuda().eval()
    act0 = activation_hook()
    act1 = activation_hook()
    act2 = activation_hook()
    net.top_model.fc[0].register_forward_hook(act0)
    net.top_model.fc[2].register_forward_hook(act1)
    net.top_model.fc[4].register_forward_hook(act2)
    test_net("TEST", net, test_dataloader)
    
    with open(f"activations/{model_name}/l0.npy", "wb") as fout:
        np.save(fout, act0.get_act())

    with open(f"activations/{model_name}/l1.npy", "wb") as fout:
        np.save(fout, act1.get_act())

    with open(f"activations/{model_name}/l2.npy", "wb") as fout:
        np.save(fout, act2.get_act())


def acs_predict():

    dataset_dir = "../../datasets/acs-20210530-gold"

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )

    net = EndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc")
    net.load_state_dict(torch.load("../../weights/chemprot-cls-end-to-end/3layer-e2e-2"))
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

def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ### Basic settings ###
    # whether to evaluate only: True (default), False
    parser.add_argument('--evaluation', default='True', choices=['True', 'False'])

    ### Training settings ###
    # mini-batch size for training: 16 (default)
    parser.add_argument('--batch_size', default=32, type=int)
    # max seq length for training: 128 (default)
    parser.add_argument('--max_seq_len', default=128, type=int)

    return parser

if __name__ == "__main__":

    # 5e-5 to 1e-5 early stopping
    # top layer pooling
    parser = get_parser()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    
    args = parser.parse_args()
    print(args)

    # acs_predict()
    train_cls_end_to_end(args.batch_size, args.max_seq_len)
    # train_cls_top_model()
