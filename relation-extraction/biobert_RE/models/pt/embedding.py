from os.path import join

import torch
import h5py
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

from datasets import *

if __name__ == "__main__":

    # parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chemprot_dir = "../../datasets/CHEMPROT"
    dataset_name = "dev"
    input_path = join(chemprot_dir, f"{dataset_name}.tsv")
    output_path = join(chemprot_dir, f"{dataset_name}.h5")
    weights_dir = "../../weights/biobert-pt-v1.0-pubmed-pmc/"
    label_filter = ["CPR:3", "CPR:4", "CPR:9", "false"]
    max_seq_len = 128
    batch_size = 32
    num_workers = 8
    max_xbuffer_size = 4 * 1024 * 1024 * 1024 # 4GB, one sameple 786,432 bytes
    shuffle = False
    tokenizer = BertTokenizer(
        join(weights_dir, "vocab.txt"),
        do_lower_case=False
    )
    bert = BertModel.from_pretrained(weights_dir)
    bert.to(device)

    dataset = ChemprotTsvDataset(tokenizer, label_filter, max_seq_len)
    dataset.load(input_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=chemprot_tsv_collate_fn
    )

    # save the hidden in a h5 file
    fout = h5py.File(output_path, "w-")

    # use buffer to store hidden in memory first
    xbuffer = []
    xbuffer_size = 0
    ybuffer = []
    part_counter = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader)):
            
            # compute hidden
            x = {k: v.to(device) for k, v in x.items()}
            output = bert(**x, return_dict=True)["last_hidden_state"]
            output = output.cpu()
            output_size = output.element_size() * output.nelement()
            xbuffer.append(output)
            xbuffer_size += output_size
            ybuffer.append(y)

            # dump buffer to file
            if xbuffer_size >= max_xbuffer_size:
                xpart = torch.cat(xbuffer, 0)
                ypart = torch.cat(ybuffer, 0)
                print(f"part {part_counter}, x: {xpart.size()} y: {ypart.size()}")
                fout.create_dataset(f"x{part_counter}", data=xpart.numpy())
                fout.create_dataset(f"y{part_counter}", data=ypart.numpy())
                xbuffer, ybuffer = [], []
                xbuffer_size = 0
                part_counter += 1

    # if there is anything in the buffer, flush it
    xpart = torch.cat(xbuffer, 0)
    ypart = torch.cat(ybuffer, 0)
    print(f"part {part_counter}, x: {xpart.size()} y: {ypart.size()}")
    fout.create_dataset(f"x{part_counter}", data=xpart.numpy())
    fout.create_dataset(f"y{part_counter}", data=ypart.numpy())

    # write metadata
    fout.create_dataset(f"num_parts", data=(part_counter + 1))

    fout.close()