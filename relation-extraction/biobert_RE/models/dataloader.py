import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from utils import utils
from utils import cpr

class TsvDataset(Dataset):
    
    def __init__(self, data_filename, tokenizer, label_map, max_seq_len, balance_data):
        super(TsvDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.label_map = label_map # label to idx map
        self.max_seq_len = max_seq_len
        # the data will be grouped by classes
        # useful for stratified sampling
        self.data = [[] for _ in range(len(label_map))]
        raw = utils.read_tsv(data_filename)
        for row in tqdm(raw, desc='tokenizing dataset'):
            input_id, y, x, original_sent = row

            # only the samples in label map are used
            if y in self.label_map:
                x = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_seq_len)
                y = self.label_map[y]
                self.data[y].append((x, y, input_id, original_sent))

        # use stratified sampling if specificed
        self.balance_data = balance_data
        self.set_size = [len(l) for l in self.data]
        self.largest_set_size = max(self.set_size)
        self.actual_data_size = sum(self.set_size)
        
    def __len__(self):
        if self.balance_data:
            return self.largest_set_size * len(self.data)
        else:
            return self.actual_data_size
    
    def __getitem__(self, index):
        
        # if data balancing is enabled, we want to return samples from each class in equal probability
        # here we partitioned the index space into equal slides and map each slides to each class
        # note that __len__ in this case is the largest class size times the number of classes
        if self.balance_data:
            label = index // self.largest_set_size
            index = index % self.largest_set_size
            x, y, _, _ = self.data[label][index % self.set_size[label]]
            return x, y
        # if data balancing is disabled, we index the data set as expected
        else:
            for label, size in enumerate(self.set_size):
                if index >= size:
                    index -= size
                else:
                    x, y, _, _ = self.data[label][index]
                    return x, y
            assert False, 'index out of bound'

# bundle individual samples into one batch
# items is a list of pairs, first element is x and the second element is y
def tsv_collate_fn(items):
    batch_x = {k: [] for k in items[0][0]} # this reads the keys in the first x
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

    # batch_x: {attention_mask, input_ids, token_type_ids}
    # batch_y: list of label
    return batch_x, batch_y

def get_train_valid(train_data_filename, valid_data_filename, vocab_filename, label_map, max_seq_len, 
                    balance_data, train_p=0.8, batch_size=4, valid_batch_size=4, num_workers=1):
    tokenizer = BertTokenizer(vocab_filename, do_lower_case=False)
    train_dataset = TsvDataset(
        data_filename=train_data_filename,
        tokenizer=tokenizer,
        label_map=label_map,
        max_seq_len=max_seq_len,
        balance_data=balance_data
    )
    valid_dataset = None

    # if validation data is not specified, randomly split the train dataset
    if valid_data_filename is None:
        total_size = len(train_dataset)
        train_size = int(total_size * train_p)
        valid_size = total_size - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    # if validation data is specified, load it directly
    else:
        valid_dataset = TsvDataset(
            data_filename=valid_data_filename,
            tokenizer=tokenizer,
            label_map=label_map,
            max_seq_len=max_seq_len,
            balance_data=False # should always be false, we are not balancing validation data
        )

    print(f'dataset loading finished: train={len(train_dataset)}, valid={len(valid_dataset)}')

    # create train validation dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=tsv_collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=tsv_collate_fn
    )
    return train_dataloader, valid_dataloader

class ACSDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_len):
        super(ACSDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw = utils.read_tsv(data_path)[1:] # remove header
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

def get_acs_inference(data_filename, vocab_filename, max_seq_len, batch_size):
    tokenizer = BertTokenizer(vocab_filename, do_lower_case=False)
    dataset = ACSDataset(
        data_path=data_filename, 
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=acs_collate_fn
    )
    return dataloader
