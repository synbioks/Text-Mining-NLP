import numpy as np
np.set_printoptions(suppress=True)
import getopt
import sys
import os

from dataset import ChemprotDataset, chemprot_collate_fn
from model import BertRE, BiLSTMTopModel, FirstTokenPoolingTopModel, HiddenPoolingTopModel

from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def run_train(name, net, train_dataset, valid_dataset, max_step, max_valid_sample, valid_every, ckpt_dir, init_step):

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        collate_fn=chemprot_collate_fn
    )

    # setup training
    print(f"Running train {name}")
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    train_step_count = 0
    epoch_count = 0

    # train loop
    while True:

        print(f"Running epoch {epoch_count}")
        for _, (x_batch, y_batch) in enumerate(train_dataloader):
            
            # decide to break
            if train_step_count >= max_step:
                return

            optimizer.zero_grad()

            x_batch = {k: v.cuda() for k, v in x_batch.items()}
            y_batch = y_batch.cuda()

            output = net(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_step_count += 1

            if valid_dataset is not None and train_step_count % valid_every == 0:
                net.eval()
                run_test("TRAIN", net, train_dataset, max_valid_sample)
                run_test("VALIDATION", net, valid_dataset, max_valid_sample)
                if ckpt_dir is not None:
                    ckpt_path = os.path.join(ckpt_dir, f"model-{init_step + train_step_count}")
                    torch.save(net.state_dict(), ckpt_path)
                net.train()
                print(f"Resume epoch {epoch_count}")
        epoch_count += 1


def run_test(name, net, test_dataset, max_sample):

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        num_workers=2,
        shuffle=True,
        collate_fn=chemprot_collate_fn
    )

    # setup testing
    print(f"Running test {name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = len(test_dataset.labels)
    confusion_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(test_dataloader):
            
            if num_tested >= max_sample:
                break

            x_batch = {k: v.cuda() for k, v in x_batch.items()}
            y_batch = y_batch.numpy()

            output = net(x_batch)
            pred = torch.argmax(F.softmax(output, dim=-1), dim=-1).cpu().numpy()

            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)

            for p, y in zip(pred, y_batch):
                confusion_mat[p][y] += 1

    print("---Results---")
    print("Total tested:", np.sum(confusion_mat))
    print("ACC:", num_correct / num_tested)
    print(confusion_mat)
                

if __name__ == "__main__":

    # arg parsing
    pretrained_weights_dir = "../../weights/biobert-pt-v1.0-pubmed-pmc/"
    init_state_path = None
    ckpt_dir = None
    do_train = True
    train_step = 10000
    init_step = 0
    train_val_freq = 1000
    num_val_sample = 1000
    num_test_sample = float("inf")
    max_seq_len = 128
    dataset_dir = "../../datasets/CHEMPROT"

    opts, args = getopt.getopt(sys.argv[1:], "", [
        "init_state=", "ckpt_dir=", "do_train=", "train_step=", "num_test_sample=", "train_val_freq=", "data_dir=", "pretrained_weights_dir="
    ])
    for opt, arg in opts:
        if opt == "--init_state":
            init_state_path = arg
            state_filename = init_state_path.split("/")[-1]
            if "-" in state_filename:
                init_step = int(state_filename.split("-")[-1])
        elif opt == "--ckpt_dir":
            ckpt_dir = arg
        elif opt == "--do_train":
            do_train = arg == "True"
        elif opt == "--train_step":
            train_step = int(arg)
        elif opt == "--num_test_sample":
            num_test_sample = int(arg)
        elif opt == "--train_val_freq":
            train_val_freq = int(arg)
        elif opt == "--data_dir":
            dataset_dir = arg
        elif opt == "--pretrained_weights_dir":
            pretrained_weights_dir = arg

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )

    # create dataloaders
    train_data = ChemprotDataset(
        data_path=os.path.join(dataset_dir, "train.tsv"), 
        tokenizer=tokenizer,
        data_balance=False,
        max_seq_len=max_seq_len
    )

    # the original tf implementation of this model is using the dev dataset for evaluation
    test_data = ChemprotDataset(
        data_path=os.path.join(dataset_dir, "dev.tsv"), 
        tokenizer=tokenizer,
        data_balance=False,
        max_seq_len=max_seq_len
    )

    valid_data = ChemprotDataset(
        data_path=os.path.join(dataset_dir, "dev.tsv"), 
        tokenizer=tokenizer,
        data_balance=False,
        max_seq_len=max_seq_len
    )

    # init model
    top_model = FirstTokenPoolingTopModel()
    net = BertRE(pretrained_weights_dir, top_model)
    if init_state_path is not None:
        net.load_state_dict(torch.load(init_state_path))
    net = net.cuda()

    if do_train:
        run_train("TRAIN", net, train_data, valid_data, train_step, num_val_sample, train_val_freq, ckpt_dir, init_step)

    if ckpt_dir is not None:
        torch.save(net.state_dict(), os.path.join(ckpt_dir, f"model-{train_step + init_step}"))

    run_test("TEST", net, test_data, num_test_sample)