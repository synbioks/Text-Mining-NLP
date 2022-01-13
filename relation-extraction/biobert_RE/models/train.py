import os
from random import sample
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.set_printoptions(suppress=True, linewidth=120)
import argparse
from tqdm import tqdm

from models.nets import get_end_to_end_net
from models.dataloader import get_train_valid
from models.optimizer import VarLROptim

from utils import cpr

def train_net(task_name, net, train_dataloader, valid_dataloader, loss_fn, optimizer, args):

    # setup training
    print(f'Running train {task_name}')
    net.train()
    train_step_count = args.resume_from_step
    epoch_count = 0
    sample_counter = 0 # number of sample seen since last optimizer update

    # train loop
    for epoch_count in range(args.epochs):
        print(f'Begin epoch {epoch_count}')
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            
            # net.train_step(x, y, loss_fn, optimizer)
            x = {k: v.cuda() for k, v in x.items()}
            y = y.cuda()
            out = net(x)
            loss = loss_fn(out, y)
            loss.backward()
            sample_counter += args.batch_size

            # gradient accumulation
            # if ga is not enabled, it will just be set equal to batch_size
            # resulting in updates after every batch
            if sample_counter >= args.ga_batch_size:
                optimizer.step()
                optimizer.zero_grad()
                sample_counter = 0
                train_step_count += 1

                # do validation and save model
                if train_step_count > args.resume_from_step and train_step_count % args.valid_freq == 0:
                    net.eval()
                    print(f'\nStep {train_step_count} finished')
                    test_net('TRAIN', net, train_dataloader, limit=100)
                    test_net('VALIDATION', net, valid_dataloader, limit=100)
                    if args.ckpt_dir is not None:
                        ckpt_path = os.path.join(args.ckpt_dir, f'{train_step_count}')
                        torch.save(net.state_dict(), ckpt_path) 
                    net.train()
                    print(f'Resume epoch {epoch_count}')
    
    # final test
    print('Train finished')
    test_net('VALIDATION', net, valid_dataloader)
    if args.ckpt_dir is not None:
        ckpt_path = os.path.join(args.ckpt_dir, f'{train_step_count}')
        torch.save(net.state_dict(), ckpt_path)

def test_net(task_name, net, test_dataloader, limit=None):

    # setup testing
    print(f"Running test {task_name}")
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = len(cpr.cpr_label_id)
    confusion_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(tqdm(test_dataloader)):
            
            # stop after a certain number of batch to save time
            if limit and i >= limit:
                break

            pred = net.predict(x_batch).cpu().numpy()

            y_batch = y_batch.numpy()
            num_tested += len(y_batch)
            num_correct += np.sum(pred == y_batch)

            for p, y in zip(pred, y_batch):
                confusion_mat[y][p] += 1

    recalls = []
    precisions = []
    f1s = []
    for i in range(num_classes):
        recall = confusion_mat[i][i] / np.sum(confusion_mat[i])
        precision = confusion_mat[i][i] / np.sum(confusion_mat[:, i])
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(2 * recall * precision / (recall + precision))

    print('---Results---')
    print('Total tested:', np.sum(confusion_mat))
    print('ACC:', num_correct / num_tested)
    print('Precision:', precisions)
    print('Recall:', recalls)
    print('F1 Scores', f1s)
    print('Confusion Matrix:')
    print(confusion_mat)


# for the reason of why we are not using type=bool in add_argument:
# see https://docs.python.org/3/library/argparse.html#type
def bool_string(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        assert False, f'invalid bool string {s}'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--valid-freq', type=int, default=1000)
    parser.add_argument('--warm-up', type=int, default=3000)
    # when training end to end, lr become the variable learning rate
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--valid-batch-size', type=int)
    parser.add_argument('--ga-batch-size', type=int) # only update parameter after seeing these many samples
    parser.add_argument('--dataloader-workers', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--bert-state-path', type=str, default='weights/biobert_large_v1.1_pubmed_torch/')
    parser.add_argument('--use-bert-large', type=bool_string, default='True')
    parser.add_argument('--top-hidden-size', type=int, default=1024)
    parser.add_argument('--top-hidden-layers', type=int, default=2)
    parser.add_argument('--freeze-bert', type=bool_string, default='False')
    parser.add_argument('--resume-from-ckpt', type=str)
    parser.add_argument('--resume-from-step', type=int, default=0)
    parser.add_argument('--train-data', type=str, default='data/merged/training/merged.txt')
    parser.add_argument('--ckpt-dir', type=str, default='D:/tmp_weight_dir')
    args = parser.parse_args()

    # calculate gradient accumulation parameter
    # if ga_batch_size is not specified, it will be set equal to batch_size
    # resulting in updates after every batch
    if args.ga_batch_size is None:
        args.ga_batch_size = args.batch_size

    # set valid batch size
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size

    print('Arguments:')
    print(args)

    # initialize top model
    bert_hidden_size = 1024 if args.use_bert_large else 768
    out_size = len(cpr.cpr_label_id) # calculate the output size
    net = get_end_to_end_net(args.bert_state_path, bert_hidden_size, out_size, args)
    if args.resume_from_ckpt is not None:
        print(f'Loading existing checkpoint: {args.resume_from_ckpt}')
        net.load_state_dict(torch.load(args.resume_from_ckpt))
    if args.freeze_bert:
        print(f'Freezing BERT weights')
        net.bert_grad_required(False)
    print('Top model structure:')
    print(net.top_model)

    # init datasets
    train_dataloader, valid_dataloader = get_train_valid(
        args.train_data,
        os.path.join(args.bert_state_path, 'vocab.txt'),
        cpr.cpr_label_id,
        args.seq_len,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.dataloader_workers
    )

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # lr this line doesn't matter because we are going to adjust it using variable learning rate
    adam = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    optimizer = adam
    # only use var lr with bert
    if not args.freeze_bert:
        optimizer = VarLROptim(adam, args.warm_up, args.lr, args.resume_from_step)
    
    train_net(
        'TRAIN', 
        net, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn,
        optimizer,
        args
    )