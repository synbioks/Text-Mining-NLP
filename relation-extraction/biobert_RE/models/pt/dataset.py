import csv
import torch
import random

from torch.utils.data import Dataset

class ChemprotDataset(Dataset):

    def __init__(self, data_path, tokenizer, data_balance=False, max_seq_len=128):
        super(ChemprotDataset, self).__init__()
        self.labels = ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        raw = []
        with open(data_path, "r", encoding="utf8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                raw.append(row)
        raw = raw[1:]

        # balance data samples by sample everyting to the size of biggest class
        classes = {k: [] for k in self.labels}
        for line in raw:
            classes[line[4]].append(line[2])
        if data_balance:
            max_class_len = max([len(v) for v in classes.values()])
            for k in classes.keys():
                class_len = len(classes[k])
                count = max_class_len - class_len
                for _ in range(count):
                    idx = random.randint(0, class_len - 1)
                    classes[k].append(classes[k][idx])
        
        self.data = []
        for label, sents in classes.items():
            for sent in sents:
                self.data.append((sent, label))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # give one sample, we use biobert's vocab to tokenize the sentence
        sent, label = self.data[index]
        x = self.tokenizer(sent, padding="max_length", truncation=True, max_length=self.max_seq_len)
        y = self.labels.index(label)

        # 137 is the "at" sign, which is used for indicating the start of an entity
        # this part is for the experiment of adapting crf to RE
        try:
            ent_pos = x["input_ids"].index(137)
        except ValueError:
            ent_pos = 999
        z = {}
        z["tags"] = [y if i >= ent_pos else 0 for i in range(len(x["input_ids"]))]
        z["ent_pos"] = ent_pos

        # x is {input_ids: [...], attention_mask: [...], token_type_ids: [...]}
        # y is a vector of label ids [...]
        return x, y, z

class ACSDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_len=128):
        super(ACSDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw = []
        with open(data_path, "r", encoding="utf8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                raw.append(row)
        raw = raw[1:]
        self.data = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id1, id2, sent = self.data[index]
        x = self.tokenizer(sent, padding="max_length", truncation=True, max_length=self.max_seq_len)
        return id1, id2, x

def chemprot_collate_fn(items):

    # this function tells dataloader how to combine individual samples into a minibatch
    # the input items are a list of individual samples
    x_batch = {k: [] for k in items[0][0]}
    y_batch = []
    extra = {k: [] for k in items[0][2]}
    for x, y, z in items:
        for k, v in x.items():
            x_batch[k].append(v)
        for k, v in z.items():
            extra[k].append(v)
        y_batch.append(y)
    x_batch = {k: torch.FloatTensor(v) if k == "attention_mask" else torch.LongTensor(v) for k, v in x_batch.items()}
    y_batch = torch.LongTensor(y_batch)
    return x_batch, y_batch, extra

def acs_collate_fn(items):
    x_batch = {k: [] for k in items[0][2]}
    id1s = []
    id2s = []
    for id1, id2, x in items:
        for k, v in x.items():
            x_batch[k].append(v)
        id1s.append(id1)
        id2s.append(id2)
    x_batch = {k: torch.FloatTensor(v) if k == "attention_mask" else torch.LongTensor(v) for k, v in x_batch.items()}
    return id1s, id2s, x_batch