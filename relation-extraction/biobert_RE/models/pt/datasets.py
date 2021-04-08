import torch
import csv
import h5py
import numpy as np

from torch.utils.data import Dataset

class ACSDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_len):
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

class ChemprotTsvDataset(Dataset):
    
    def __init__(self, tokenizer, label_filter, max_seq_len):
        super(ChemprotTsvDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_filter = label_filter
        self.max_seq_len = max_seq_len
    
    def load(self, data_path):
        
        raw = []
        with open(data_path, "r", encoding="utf8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                raw.append(row)
        raw = raw[1:]
        
        self.data = []
        for line in raw:
            x = line[2]
            y = line[4]
            if y in self.label_filter:
                self.data.append((x, y))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index]
        x = self.tokenizer(x, padding="max_length", truncation=True, max_length=self.max_seq_len)
        y = self.label_filter.index(y)
        return x, y

# bundle individual samples into one batch
def chemprot_tsv_collate_fn(items):
    batch_x = {k: [] for k in items[0][0]}
    batch_y = []
    for x, y in items:
        for k, v in x.items():
            batch_x[k].append(v)
        batch_y.append(y)
    batch_x = {
        k: torch.tensor(v, dtype=torch.double) 
        if k == "attention_mask" 
        else torch.tensor(v, dtype=torch.long) 
        for k, v in batch_x.items()
    }
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    return batch_x, batch_y

class ChemprotH5Dataset(Dataset):

    def __init__(self):
        super(ChemprotH5Dataset, self).__init__()
    
    def load(self, data_path):
        self.xdata = []
        self.ydata = []
        fin = h5py.File(data_path, "r")
        num_parts = np.array(fin.get("num_parts")) + 0 # hack the type to int
        for i in range(num_parts):
            self.xdata.append(torch.tensor(np.array(fin.get(f"x{i}")), dtype=torch.float))
            self.ydata.append(torch.tensor(np.array(fin.get(f"y{i}")), dtype=torch.long))
        self.xdata = torch.cat(self.xdata, 0)
        self.ydata = torch.cat(self.ydata, 0)
    
    def __len__(self):
        return self.ydata.size(0)
    
    def __getitem__(self, index):
        return self.xdata[index], self.ydata[index]

def chemprot_h5_collate_fn(items):
    batch_x = []
    batch_y = []
    for x, y in items:
        batch_x.append(x.unsqueeze(0))
        batch_y.append(y)
    batch_x = torch.cat(batch_x, 0)
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    return batch_x, batch_y