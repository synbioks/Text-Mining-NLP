import numpy as np
np.set_printoptions(suppress=True)
import getopt
import sys, uuid, random
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
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import apex

from datetime import datetime
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from drugprot_evln.src.main import main as evln_main
from drugprot_evln.src.main import parse_arguments as evln_parseargs

CKPT_ROOT = "../../weights"
DATASET_ROOT = "../../datasets"
CHEMPROT_ROOT = os.path.join(DATASET_ROOT, "CHEMPROT")
logfile = None

def logprint(t):
    logfile.write(str(t)); logfile.write("\n");
    print(t)

def train_net(task_name, net, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, 
              n_epochs, ckpt_dir, init_step, label_filter, n_its=1):

    ckpt_dir += "/model"
    # setup training
    logprint(f"Running train {task_name}")
    net.train()
    train_step_count = 0
    epoch_count = 0
    all_val_f1 = []
    best_f1 = 0

    # train loop
    for epoch_count in range(n_epochs):
        logprint(f"Running epoch {epoch_count}\n")
        optimizer.zero_grad()
        for (btch_no, (_, x, y, loc_ids)) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # print(btch_no)
            if type(loss_fn) == nn.modules.loss.BCEWithLogitsLoss:
                y = nn.functional.one_hot(y).float()
            if btch_no % n_its == n_its-1:
                optimizer.zero_grad()
            net.train_step_new(x, y, loc_ids, loss_fn, n_its)
            if btch_no % n_its == n_its-1:
                optimizer.step()
                scheduler.step()
        
        net.eval()
        logprint(f"Step {init_step + train_step_count} finished")
        test_net_drugprot("TRAINING", net, train_dataloader, label_filter)
        val_f1 = test_net_drugprot("DEVELOPMENT", net, valid_dataloader, label_filter)
        all_val_f1.append(val_f1)
        if ckpt_dir is not None and val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(net, ckpt_dir)
        net.train()

    if ckpt_dir and os.path.exists(ckpt_dir):          
        net = torch.load(ckpt_dir)
    logprint("Epoch wise f1 val score three classes : {}".format(str(all_val_f1)))
    val_f1 = test_net_drugprot("DEVELOPMENT", net, valid_dataloader, label_filter)
    return net

def test_net(task_name, net, test_dataloader, target_names):

    # setup testing
    print(f"Running test {task_name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = 14
    confusion_mat = np.zeros((num_classes, num_classes))
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (minfo, x_batch, y_batch) in tqdm(enumerate(test_dataloader)):

            pred = net.predict(x_batch).cpu().numpy()

            y_batch = y_batch.numpy()
            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)

            for p, y in zip(pred, y_batch):
                confusion_mat[p][y] += 1
            y_true.extend(y_batch.tolist())
            y_pred.extend(pred.tolist())

    recalls = []
    precisions = []
    f1s = []

    print("---Results---")
    print("Total tested: {}".format(np.sum(confusion_mat)))
    print("ACC: {}".format(num_correct / num_tested))
    print(classification_report(y_true, y_pred, target_names=target_names))
    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    print(confusion_matrix([target_names[lbl] for lbl in y_true], 
                           [target_names[lbl] for lbl in y_pred], labels=target_names))
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    print("{} Weighted f1 score : {}".format(task_name, f1_weighted))
    print("{} Macro    f1 score : {}".format(task_name, f1_macro))
    print("{} Micro    f1 score : {}".format(task_name, f1_micro))

    return np.mean(f1_macro)

def test_net_drugprot(task_name, net, test_dataloader, target_names, ckpt_dir=None):
    # setup testing
    logprint(f"Running test on {task_name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = 14
    y_true, y_pred = [], []
    allpreds = []
    with torch.no_grad():
        for i, (minfo_batch, x_batch, y_batch, loc_ids) in tqdm(enumerate(test_dataloader)):
            pred = net.predict(x_batch, loc_ids).cpu().numpy()
            y_batch = y_batch.numpy()
            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)
            for minfo, y_ in zip(minfo_batch, pred.tolist()):
                if target_names[y_] != "NA":
                    allpreds.append([minfo[0], target_names[y_], minfo[1], minfo[2]])

            y_true.extend(y_batch.tolist())
            y_pred.extend(pred.tolist())

    allpreds_df = pd.DataFrame(allpreds)
    nnn = random.choice([i for i in range(100)])
    base_pth = "/tmp"
    if ckpt_dir is not None:
        base_pth = ckpt_dir
    tmppath     = "{}/random_{}_{}.csv".format(base_pth, task_name.lower(), nnn)
    allpreds_df.to_csv(tmppath, sep="\t", index=False, header=False)
    pmids       = set([t[0] for t in allpreds])
    tmppath2    = "{}/pmids_{}_{}.txt".format(base_pth, task_name.lower(), nnn)
    if len(allpreds) == 0:
        return 0

    logprint("Paths for writing the information : {} {}".format(tmppath, tmppath2))

    if task_name == "TEST":
        return


    with open(tmppath2, "w") as f:
        f.write("\n".join(list(pmids)))

    eval_args = evln_parseargs()
    eval_args.gs_path   = "../../datasets/drugprot-gs-training-development/{}/drugprot_{}_relations.tsv".format(task_name.lower(), task_name.lower())
    eval_args.ent_path  = "../../datasets/drugprot-gs-training-development/{}/drugprot_{}_entities.tsv".format(task_name.lower(), task_name.lower())
    eval_args.pred_path = tmppath
    eval_args.pmids     = tmppath2
    eval_args.split     = task_name
    logprint("Running evaluation with the official script ... ")
    f1_official = evln_main(eval_args)

    logprint("---Results---")
    logprint("ACC: {}".format(num_correct / num_tested))
    # logprint(classification_report(y_true, y_pred, labels=target_names))
    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    logprint(confusion_matrix([target_names[lbl] for lbl in y_true], [target_names[lbl] for lbl in y_pred], labels=target_names))
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    logprint("{} Weighted f1 score : {}".format(task_name, f1_weighted))
    logprint("{} Macro    f1 score : {}".format(task_name, f1_macro))
    logprint("{} Micro    f1 score : {}".format(task_name, f1_micro))

    return f1_official

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

# def train_cls_end_to_end(args, batch_size=32, max_seq_len=128, epochs=10):
def train_cls_end_to_end(args, pth, n_its, datasetname = "CHEMPROT", ckpt_dir=None):
    batch_size  = args.batch_size
    max_seq_len = args.max_seq_len
    epochs      = args.epochs
    upsampling  = args.upsampling

    logprint("Running the model for ... {} dataset".format(datasetname))
    init_state = None
    top_model_init_state = None

    # net
    init_step = 0

    # tokenizer and model net
    if args.use_bert_large:
        net = EndToEnd("../../weights/biobert_large", top_model=CLSTopModel(datasetname))
        tokenizer = BertTokenizer(
            # "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
            "../../weights/biobert_large/vocab_cased_pubmed_pmc_30k.txt",
            do_lower_case=False
        )
    else:
        net = EndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc", use_loc_ids=args.use_loc_ids, 
                    top_model=CLSTopModel(datasetname, layers=int(args.layers), use_loc_ids=args.use_loc_ids))
        tokenizer = BertTokenizer(
            "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
            do_lower_case=False
        )

    label_filter = ["CPR:3", "CPR:4", "CPR:9", "false"]

    # dataloaders
    train_dataloader, valid_dataloader, test_dataloader, label_filter, wts = get_dataloaders(
        datasetname, tokenizer, max_seq_len, batch_size, args.upsampling)

    if ckpt_dir is not "":
        net = torch.load(ckpt_dir + "/model")
        test_net_drugprot("DEVELOPMENT", net, valid_dataloader, label_filter, ckpt_dir)
        test_net_drugprot("TEST", net, test_dataloader, label_filter, ckpt_dir)
        return

    net.clip_param_grad = 1.0
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
    if args.weighted:
        logprint(wts)
        loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(wts).cuda())
    if args.use_sigmoid:
        loss_fn = nn.BCEWithLogitsLoss()

    # Prepare optimizer
    t_total = len(train_dataloader) * epochs / n_its
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    logprint("Creating optimizer with learning rate : {}".format(float(args.learning_rate)))
    optimizer = optim.AdamW(optimizer_grouped_parameters,
                            lr=float(args.learning_rate), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)
    logprint(net)

    logprint("Dataset statistics : ")
    logprint("Train / Val / Test : {} / {} / {}".format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))
    logprint("Training model with params bs/seqlen : {} {}".format(batch_size, max_seq_len))

    freeze_pretrain = int(args.freeze_pretrain)
    if freeze_pretrain:
        # freeze the weights
        for param in net.bert.parameters():
            param.requires_grad = False

        # need to create a separate optimizer and scheduler for this stage.
        t_total = len(train_dataloader) * freeze_pretrain
        optimizer_n = optim.AdamW(optimizer_grouped_parameters, lr=0.001, eps=1e-8)
        scheduler_n = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)

        net = train_net(
            "TRAIN", 
            net, 
            train_dataloader, 
            valid_dataloader, 
            loss_fn, 
            optimizer_n, 
            scheduler_n,
            freeze_pretrain,  # n epochs
            pth, # checkpoint directory
            init_step,
            label_filter
        ) 

        # Unfreeze the weights
        for param in net.bert.parameters():
            param.requires_grad = True

    net.half()

    # train / test loop
    net = train_net(
        "TRAIN", 
        net, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn, 
        optimizer, 
        scheduler,
        epochs,  # n epochs
        pth, # checkpoint directory
        init_step,
        label_filter,
        n_its
    ) 

    logprint("\n\n\n\nRunning final evaluaton .... ")
    test_net_drugprot("DEVELOPMENT", net, valid_dataloader, label_filter)
    test_net_drugprot("TEST", net, test_dataloader, label_filter, ckpt_dir=pth)


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


def acs_predict(max_seq_len=128, wts_path="../../weights/chemprot-cls-end-to-end/3layer-e2e-2"):

    print("Predicting on ACS with seq length {} model checkpoint : {}".format(max_seq_len, wts_path))
    dataset_dir = "../../datasets/acs-20210530-gold"

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )

    net = EndToEnd("../../weights/biobert-pt-v1.0-pubmed-pmc")
    # net.load_state_dict(torch.load(wts_path))
    net = torch.load(wts_path)
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
        with open(os.path.join(article_dir, "re_output_{}.tsv".format(max_seq_len)), "w", encoding="utf8") as fout:
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
    # upsampling for training: 128 (default)
    parser.add_argument('--upsampling', default=1, type=int)
    # max seq length for training: 128 (default)
    parser.add_argument('--max_seq_len', default=128, type=int)
    # n iters to wait before stepping the opimizer: 1 (default)
    parser.add_argument('--n_its', default=1, type=int)
    # epcohs training: 5 (default)
    parser.add_argument('--epochs', default=10, type=int)
    # whether to train using weights: True (default), False
    parser.add_argument('--weighted', action='store_true')
    # dataset name
    parser.add_argument('--datasetname', default='DRUGPROT', choices=['DRUGPROT', 'CHEMPROT'])
    # no of layers in top level model
    parser.add_argument('--layers', default='3', choices=['1', '3'])
    # learning rate of the model
    parser.add_argument('--learning_rate', default='0.00005', choices=['0.00001', '0.00002', '0.00003', '0.00004', '0.00005'])
    # Whether to freeze and pretrain
    parser.add_argument('--freeze_pretrain', default='0', choices=['0', '2'])
    # use location ids
    parser.add_argument('--use_loc_ids', action='store_true')
    # use sigmoid
    parser.add_argument('--use_sigmoid', action='store_true')
    # use bert large
    parser.add_argument('--use_bert_large', action='store_true')
    # freeze bert large init layers
    parser.add_argument('--freeze_bert_large_init_layers', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='The ckpt_dir to parse.', default="")
    
    return parser

if __name__ == "__main__":

    # 5e-5 to 1e-5 early stopping
    # top layer pooling
    parser = get_parser()
    args = parser.parse_args()
    # global logfile
    args_dict = dict(vars(args))
    # print([(k, args_dict[k]) for k in sorted(list(args_dict.keys()))])
    pth = args.ckpt_dir
    if not args.ckpt_dir:
        pth = "../../weights/{}".format(args.datasetname) + \
            "_".join(["{}_{}".format(k[:5], args_dict[k]) for k in sorted(list(args_dict.keys()))]) + \
            "_" + str(datetime.now().strftime("%d_%m_%H_%M"))
        
        os.mkdir(pth)
    logfile = open(pth + "/f_"+ str(datetime.now().strftime("%d_%m_%H_%M")) +"_.log", "w")

    logprint("Path : {}".format(pth))

    logprint("Args for the main file ..... ")
    logprint(args)
    for k in args_dict:
        print(k, args_dict[k])

    logprint("\n\n\n\n")

    train_cls_end_to_end(args, pth, args.n_its, datasetname=args.datasetname, ckpt_dir=args.ckpt_dir)

    logfile.close()

# python -u main_warmup_wd_fix.py --layers 1 --max_seq_len 256 --batch_size 2 --learning_rate 0.00003 --n_its 8
# python -u main_warmup_wd_fix.py --layers 1 --max_seq_len 256 --use_sigmoid --batch_size 16 --learning_rate 0.00001 > logs/basic_run_drugprot_layers1_msl_256_sigmoid_0.00001;
