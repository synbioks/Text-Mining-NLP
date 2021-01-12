import os
import sys
import getopt
from os.path import abspath, isdir, join
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import BertRE, FirstTokenPoolingTopModel
from transformers import BertTokenizer

from dataset import ACSDataset, acs_collate_fn

def run_predict(net, dataset):
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
        num_workers=4,
        shuffle=False,
        collate_fn=acs_collate_fn
    )

    res = []
    with torch.no_grad():
        for _, (id1s, id2s, batch_x) in enumerate(dataloader):
            batch_x = {k: v.cuda() for k, v in batch_x.items()}
            output = net(batch_x)
            score = F.softmax(output, dim=-1)
            pred = torch.argmax(score, dim=-1)
            score = score.cpu().tolist()
            pred = pred.cpu().tolist()
            for i in range(len(id1s)):
                res.append((id1s[i], id2s[i], pred[i], score[i][pred[i]]))
    return res

if __name__ == "__main__":

    # script parames
    dataset_dir = abspath("../../datasets/acs")
    # we are not actually using this pretrained weights, the init_state will override the pretrained weights
    pretrained_weights_dir = abspath("../../weights/biobert-pt-v1.0-pubmed-pmc")
    init_state_path = abspath("../../weights/biobert-pt-chemprot/model-12000")
    max_seq_len = 128

    opts, args = getopt.getopt(sys.argv[1:], "", [
        "dataset_dir=", "pretrained_weights_dir=", "init_state="
    ])
    for opt, arg in opts:
        if opt == "--dataset_dir":
            dataset_dir = arg
        elif opt == "--pretrained_weights_dir":
            pretrained_weights_dir = arg
        elif opt == "--init_state":
            init_state_path = arg

    # tokenizer
    tokenizer = BertTokenizer(
        "../../weights/biobert-pt-v1.0-pubmed-pmc/vocab.txt",
        do_lower_case=False
    )

    # init model
    top_model = FirstTokenPoolingTopModel()
    net = BertRE(pretrained_weights_dir, top_model)
    net.load_state_dict(torch.load(init_state_path))
    net = net.cuda()
    net.eval()

    for pub_num in tqdm(os.listdir(dataset_dir)):
        article_dir = join(dataset_dir, pub_num)
        assert isdir(article_dir)
        dataset = ACSDataset(
            data_path=join(article_dir, "re_input.tsv"),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )
        output = run_predict(net, dataset)
        with open(join(article_dir, "re_output.tsv"), "w", encoding="utf8") as fout:
            fout.write("id1\tid2\tclass\tconfidence\n")
            for _, (id1, id2, pred, score) in enumerate(output):
                fout.write(f"{id1}\t{id2}\t{pred}\t{score}\n")