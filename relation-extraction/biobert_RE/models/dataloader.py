from os import truncate
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from utils import utils

class DrugProtDataset(Dataset):
    
    def __init__(self, data_filename, tokenizer, label_map, max_seq_len):
        super(DrugProtDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_map = label_map # label to idx map
        self.max_seq_len = max_seq_len
        self.data = []
        raw = utils.read_tsv(data_filename)
        for row in tqdm(raw, desc='tokenizing dataset'):
            input_id, y, x, original_sent = row
            if y in self.label_map:
                x = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_seq_len)
                y = self.label_map[y]
                self.data.append((x, y, input_id, original_sent))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y, _, _ = self.data[index]
        # x = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_seq_len)
        # y = self.label_map[y]
        return x, y

# bundle individual samples into one batch
def drugprot_collate_fn(items):
    batch_x = {k: [] for k in items[0][0]}
    batch_y = []
    for x, y in items:
        for k, v in x.items():
            batch_x[k].append(v)
        batch_y.append(y)
    batch_x = {
        k: torch.tensor(v, dtype=torch.double) 
        if k == 'attention_mask' 
        else torch.tensor(v, dtype=torch.long) 
        for k, v in batch_x.items()
    }
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    return batch_x, batch_y

def get_drugprot_label_map():
    label_map = sorted([
        'SUBSTRATE_PRODUCT-OF',
        'INDIRECT-DOWNREGULATOR',
        'INHIBITOR',
        'AGONIST',
        'SUBSTRATE',
        'AGONIST-ACTIVATOR',
        'ACTIVATOR',
        'AGONIST-INHIBITOR',
        'DIRECT-REGULATOR',
        'ANTAGONIST',
        'PRODUCT-OF',
        'PART-OF',
        'INDIRECT-UPREGULATOR'
    ]) + ['NOT']
    return {k: i for i, k in enumerate(label_map)}

def get_drugprot_train_valid(data_filename, vocab_filename, label_map, max_seq_len, train_p=0.8, batch_size=4, valid_batch_size=4, num_workers=1):
    tokenizer = BertTokenizer(vocab_filename, do_lower_case=False)
    dataset = DrugProtDataset(
        data_filename=data_filename,
        tokenizer=tokenizer,
        label_map=label_map,
        max_seq_len=max_seq_len
    )
    total_size = len(dataset)
    train_size = int(total_size * train_p)
    valid_size = total_size - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=drugprot_collate_fn
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=drugprot_collate_fn
    )
    return train_dataloader, valid_dataloader

if __name__ == '__main__':
    tokenizer = BertTokenizer('weights/biobert_large_v1.1_pubmed_torch/vocab.txt', do_lower_case=False)
    label_map = sorted([
        'SUBSTRATE_PRODUCT-OF',
        'INDIRECT-DOWNREGULATOR',
        'INHIBITOR',
        'AGONIST',
        'SUBSTRATE',
        'AGONIST-ACTIVATOR',
        'ACTIVATOR',
        'AGONIST-INHIBITOR',
        'DIRECT-REGULATOR',
        'ANTAGONIST',
        'PRODUCT-OF',
        'PART-OF',
        'INDIRECT-UPREGULATOR'
    ]) + ['NOT']
    label_map = {k: i for i, k in enumerate(label_map)}
    data = DrugProtDataset(
        data_filename='data/DrugProt/training/drugprot_train.txt',
        tokenizer=tokenizer,
        label_map=label_map,
        max_seq_len=256
    )
    dataloader = DataLoader(
        dataset=data,
        batch_size=4,
        num_workers=1,
        shuffle=True,
        collate_fn=drugprot_collate_fn
    )
    for i, (x, y) in enumerate(dataloader):
        print(y, x)
        break