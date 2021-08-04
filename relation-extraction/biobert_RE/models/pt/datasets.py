import torch
import csv
import h5py
import numpy as np
import os 

from torch.utils.data import Dataset, DataLoader

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

class DrugProtDataset(Dataset):
    
    def __init__(self, tokenizer, label_filter, max_seq_len, upsampling=1):
        super(DrugProtDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_filter = label_filter
        self.max_seq_len = max_seq_len
        self.upsampling = upsampling
    
    def load(self, data_path):
        
        raw = []
        # print("buggggggggggggg here ......\n\n\n\n")
        with open(data_path, "r", encoding="utf8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for i, row in enumerate(reader):
                raw.append(row)
                # if i > 100:
                #     break
        raw = raw[1:]
        
        self.data        = []
        self.data_NA     = []
        self.data_not_NA = []
        for line in raw:
            metainfo = (line[0], line[2], line[3])
            x = line[4]
            y = line[5]
            if y in self.label_filter:
                if y == "NA":
                    self.data_NA.append((metainfo, x, y))
                else:
                    self.data_not_NA.append((metainfo, x, y))

        new_not_NA = []
        for _ in range(self.upsampling):
            new_not_NA.extend(self.data_not_NA)
        
        self.data = new_not_NA + self.data_NA
        print("Upsampling ratio : {}".format(self.upsampling))
        print("Previous / new count : {} {}".format(len(self.data_NA) + len(self.data_not_NA), len(self.data)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        minfo, x, y = self.data[index]
        x = self.tokenizer(x, padding="max_length", truncation=True, max_length=self.max_seq_len)
        y = self.label_filter.index(y)
        return minfo, x, y

# bundle individual samples into one batch
def drugprot_collate_fn(items):
    batch_x = {k: [] for k in items[0][1]}
    batch_y = []
    batch_minfo = []
    for minfo, x, y in items:
        for k, v in x.items():
            batch_x[k].append(v)
        batch_y.append(y); batch_minfo.append(minfo)
    batch_x = {
        k: torch.tensor(v, dtype=torch.double) 
        if k == "attention_mask" 
        else torch.tensor(v, dtype=torch.long) 
        for k, v in batch_x.items()
    }
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    return batch_minfo, batch_x, batch_y

def get_dataloaders(datsetname, tokenizer, max_seq_len, batch_size):
    label_filter_map = {
        "DRUGPROT" : ['PRODUCT-OF', 'ACTIVATOR', 'ANTAGONIST', 'INDIRECT-UPREGULATOR', 'INDIRECT-DOWNREGULATOR', 'SUBSTRATE', 'PART-OF', 'DIRECT-REGULATOR', 'INHIBITOR', 'AGONIST-ACTIVATOR', 'SUBSTRATE_PRODUCT-OF', 'AGONIST', 'AGONIST-INHIBITOR', 'NA'],
        "CHEMPROT" : ["CPR:3", "CPR:4", "CPR:9", "false"]
    }

    dataset_fn_map = {
        "DRUGPROT" : DrugProtDataset,
        "CHEMPROT" : ChemprotTsvDataset
    }

    collate_fn_map = {
        "DRUGPROT" : drugprot_collate_fn,
        "CHEMPROT" : chemprot_tsv_collate_fn
    }

    CHEMPROT_ROOT = "../../datasets/CHEMPROT"
    DRUGPROT_ROOT = "../../datasets/drugprot-gs-training-development/{}/re_input_all.tsv"
    files_map = {
        "CHEMPROT" : [os.path.join(CHEMPROT_ROOT, t) for t in 
                      ["train_4cls.tsv", "dev_4cls.tsv", "test_4cls.tsv"]],
        "DRUGPROT" : [DRUGPROT_ROOT.format(t) for t in ["training", "development", "development"]],
    }

    label_filter = label_filter_map[datsetname]
    dataset_fn   = dataset_fn_map[datsetname]
    files_list   = files_map[datsetname]

    train_data = dataset_fn(tokenizer, label_filter, max_seq_len, args.upsampling)
    train_data.load(files_list[0])
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn_map[datsetname]
    )

    valid_data = dataset_fn(tokenizer, label_filter, max_seq_len)
    valid_data.load(files_list[1])
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn_map[datsetname]
    )
    
    test_data = dataset_fn(tokenizer, label_filter, max_seq_len)
    test_data.load(files_list[2])
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn_map[datsetname]
    )

    return train_dataloader, valid_dataloader, test_dataloader, label_filter
