import torch
import csv
import h5py
import numpy as np
import os 
from collections import Counter

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
    
    def __init__(self, tokenizer, label_filter, max_seq_len, use_midpoint=False):
        super(ChemprotTsvDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_filter = label_filter
        self.max_seq_len = max_seq_len
        assert(not use_midpoint)
    
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
    
    def __init__(self, tokenizer, label_filter, max_seq_len, upsampling=1, use_midpoint=False):
        super(DrugProtDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_filter = label_filter
        self.max_seq_len = max_seq_len
        self.upsampling = upsampling
        self.use_midpoint = use_midpoint
    
    def load(self, data_path):
        
        raw = []
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

        if self.use_midpoint:
            minfo, x, y = self.data[index]
            x = self.tokenizer(x, padding="max_length", max_length=self.max_seq_len)
            y = self.label_filter.index(y)
            x_vals = x["input_ids"]
        
            # deal with the case when the gene_loc and the chem_loc is outside the max_seq_length.
            gene_loc = 0; chem_loc = 0
            for i  in range(len(x_vals)):
                if x_vals[i] == 137 and (x_vals[i+1] == 24890 or x_vals[i+1] == 42769):
                    chem_loc = i
                if x_vals[i] == 137 and (x_vals[i+1] == 25075 or x_vals[i+1] == 44017):
                    gene_loc = i
   
            # seq len 200
            # gene 100, chem 150
            # mid point 125 , seq inp : (125 - 128/2, 125 + 128/2)
            if len(x_vals) > self.max_seq_len:
                start_pt = (gene_loc + chem_loc) // 2 - self.max_seq_len // 2
                
                if start_pt < 0:
                    # case when start point is less than 0 implies seq len of 128 most likely covers both gene and chemical
                    start_pt = 0

                end_pt = start_pt + self.max_seq_len

                if end_pt > len(x_vals):
                    start_pt = len(x_vals) - self.max_seq_len
                    end_pt = start_pt + self.max_seq_len

                def print_issue():
                    _, in_x, in_y = self.data[index]
                    if in_y != "NA":
                        print("Found sample with tokenized length greater than max_seq_len ..... index / orig length / start pt / end pt / gene / chem / label {} / {} / {} / {} / {} / {} / {}".format(
                            index, len(x_vals), start_pt, end_pt, gene_loc, chem_loc, in_y))
                        print(in_x, in_y)
                    # print(self.data[index][1])
                    # (140 + 200) // 2 - 128 // 2 = 170-64 = 106; 106 + 128

                for k in x:
                    x[k] = x[k][start_pt:end_pt]

                if len(x["input_ids"]) != self.max_seq_len:
                    print_issue()
                    raise NotImplementedError("There is some issue with current sample")

                if gene_loc < start_pt or chem_loc < start_pt or gene_loc > end_pt or chem_loc > end_pt:
                    # print("Found a sample violating the end constraints while doing tokenization.")
                    # print("Found sample with tokenized length greater than max_seq_len ..... index / orig length / start pt / end pt / gene / chem  {} / {} / {} / {} / {} / {}".format(
                    #     index, len(x_vals), start_pt, end_pt, gene_loc, chem_loc))
                    print_issue()
                
        else:
            minfo, x, y = self.data[index]
            x = self.tokenizer(x, padding="max_length", truncation=True, max_length=self.max_seq_len)
            y = self.label_filter.index(y)
            x_vals = x["input_ids"]
        
            # deal with the case when the gene_loc and the chem_loc is outside the max_seq_length.
            gene_loc = 0; chem_loc = 0
            for i  in range(len(x_vals)):
                if x_vals[i] == 137 and (x_vals[i+1] == 24890 or x_vals[i+1] == 42769):
                    chem_loc = i
                if x_vals[i] == 137 and (x_vals[i+1] == 25075 or x_vals[i+1] == 44017):
                    gene_loc = i

            
            # tokenizer.tpkenize("@GENE") // 137, 24890 or 137, 42769

            # try:
            #     a = (minfo, x, y, [[0], [gene_loc], [chem_loc]])
            # except Exception as ex:
            #     print(ex)
            #     print(self.data[index][1])
            #     print(x_vals)

        return minfo, x, y, [[0], [gene_loc], [chem_loc]]

# bundle individual samples into one batch
def drugprot_collate_fn(items):
    batch_x = {k: [] for k in items[0][1]}
    batch_y = []
    batch_minfo = []
    batch_loc_ids = []
    for minfo, x, y, loc_ids in items:
        for k, v in x.items():
            batch_x[k].append(v)
        batch_y.append(y); batch_minfo.append(minfo); batch_loc_ids.append(loc_ids)
    batch_x = {
        k: torch.tensor(v, dtype=torch.double) 
        if k == "attention_mask" 
        else torch.tensor(v, dtype=torch.long) 
        for k, v in batch_x.items()
    }
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    batch_loc_ids = torch.tensor(batch_loc_ids, dtype=torch.long)
    return batch_minfo, batch_x, batch_y, batch_loc_ids

def get_dataloaders(datsetname, tokenizer, max_seq_len, batch_size, upsampling=1, use_midpoint=False):
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
    # DRUGPROT_ROOT = "../../datasets/drugprot-gs-training-development/{}/re_input_all.tsv"
    DRUGPROT_ROOT = "../../datasets/drugprot-gs-training-development-test/{}/re_input_all.tsv"
    files_map = {
        "CHEMPROT" : [os.path.join(CHEMPROT_ROOT, t) for t in 
                      ["train_4cls.tsv", "dev_4cls.tsv", "test_4cls.tsv"]],
        "DRUGPROT" : [DRUGPROT_ROOT.format(t) for t in ["training", "development", "test"]],
    }

    if use_midpoint:
        print("\n\nCreating datasets with midpoints .......\n\n")

    label_filter = label_filter_map[datsetname]
    dataset_fn   = dataset_fn_map[datsetname]
    files_list   = files_map[datsetname]

    lens_pkl = {}
    train_data = dataset_fn(tokenizer, label_filter, max_seq_len, upsampling, use_midpoint=use_midpoint)
    train_data.load(files_list[0])
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn_map[datsetname])
    # lens = []
    # for _, x, _ in train_data:
    #     lens.append(len(x["input_ids"]))
    # lens = np.array(lens)
    # print("train ",  np.sum(lens > 128), np.sum(lens > 256), np.sum(lens > 512), lens.shape)
    # lens_pkl["train"] = lens

    valid_data = dataset_fn(tokenizer, label_filter, max_seq_len, use_midpoint=use_midpoint)
    valid_data.load(files_list[1])
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn_map[datsetname])
    # lens = []
    # for _, x, _ in valid_data:
    #     lens.append(len(x["input_ids"]))
    # lens = np.array(lens)
    # print("development ", np.sum(lens > 128), np.sum(lens > 256), np.sum(lens > 512), lens.shape)
    # lens_pkl["dev"] = lens
    # import pickle
    # with open("../../lens.pkl", "wb") as f:
    #     pickle.dump(lens_pkl, f)
    # return

    test_data = dataset_fn(tokenizer, label_filter, max_seq_len, use_midpoint=use_midpoint)
    test_data.load(files_list[2])
    test_dataloader  = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn_map[datsetname])

    weights = [0 for _ in label_filter]
    # lbls = [t[2] for t in train_data]
    # lbls_ctr = Counter(lbls)
    # print(lbls_ctr)
    # for k in sorted(list(lbls_ctr.keys())):
    #     weights[k] = 1/lbls_ctr[k]

    # weights = np.array(weights) * max(lbls_ctr.values())
    # print(weights)
    # print(np.sqrt(weights))
    # weights = np.clip(np.sqrt(weights), 0, 10)
    # print(weights)

    return train_dataloader, valid_dataloader, test_dataloader, label_filter, weights
