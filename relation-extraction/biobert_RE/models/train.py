import os
from importlib_metadata import entry_points
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import habana_frameworks.torch.distributed.hccl
import os

import numpy as np
np.set_printoptions(suppress=True, linewidth=120)
import argparse
from tqdm import tqdm
import warnings

from models.nets import get_end_to_end_net
from models.dataloader import get_train_valid_test
from models.dataloader import get_acs_inference
from models.dataloader import get_brat_eval
from models.optimizer import VarLROptim

from utils import cpr
from utils.activation_vis import ActivationHook
from utils.early_stop import EarlyStopping
from dataset_processing.input_to_annbrat import convert_json_to_brat, input_to_re
import wandb

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
parser.add_argument('--bert-state-path', type=str, default='model_weights/biobert_large_v1.1_pubmed_torch/')
parser.add_argument('--use-bert-large', type=bool_string, default='True')
parser.add_argument('--top-hidden-size', nargs='*', default=[1024, 1024])
parser.add_argument('--freeze-bert', type=bool_string, default='False')
parser.add_argument('--resume-from-ckpt', type=str)
parser.add_argument('--resume-from-step', type=int, default=0)
parser.add_argument('--train-data', type=str, default='data/merged/training/train.txt')
parser.add_argument('--valid-data', type=str, default='data/merged/training/vali.txt')
parser.add_argument('--test-data', type=str, default='data/merged/dev/merged.txt')
parser.add_argument('--label-map-name', type=str, default='merged')
parser.add_argument('--inference-data', type=str, default='data/acs-data')
parser.add_argument('--ckpt-dir', type=str, default=None)
parser.add_argument('--balance-dataset', type=bool_string, default='False')
parser.add_argument('--do-train', type=bool_string, default='True')
parser.add_argument('--do-inference', type=bool_string, default='False')
parser.add_argument('--do-brateval', type=bool_string, default='False')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--record-activation', nargs='+', default=[])
parser.add_argument('--record-wandb', type=str, default='')
parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.') 
parser.add_argument('--dataloader_workers', type=int, default=2) 
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

def main(rank:int, world_size, args):
    
    ddp_setup(rank, world_size, args.dist_backend)
     # initialize top model
    bert_hidden_size = 1024 if args.use_bert_large else 768
    out_size = len(cpr.get_label_map()) # calculate the output size
    net = get_end_to_end_net(args.bert_state_path, bert_hidden_size, args.top_hidden_size, out_size, args.activation).to(args.device)
    net = DDP(net, device_ids=[args.local_rank])
    if args.resume_from_ckpt is not None:
        print(f'Loading existing checkpoint: {args.resume_from_ckpt}')
        net.module.load_state_dict(torch.load(args.resume_from_ckpt))
    if args.freeze_bert:
        print(f'Freezing BERT weights')
        net.module.bert_grad_required(False)
    print('Top model structure:')
    print(net.module.top_model)

    # init datasets
    # if validation data is not specified, we will do a train/valid 80-20 split on the train data
    train_dataloader, valid_dataloader, test_dataloader = get_train_valid_test(
        args.train_data,
        args.valid_data,
        args.test_data,
        os.path.join(args.bert_state_path, 'vocab.txt'),
        cpr.get_label_map(),
        args.seq_len,
        balance_data=args.balance_dataset,
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
    
    if args.do_train:
        train_net(
            'TRAIN', 
            net, 
            train_dataloader, 
            valid_dataloader, 
            test_dataloader,
            loss_fn,
            optimizer,
            args
        )
    
    if args.do_inference:
        inference_net(
            'INFERENCE',
            net,
            args
        )
    
    if args.do_brateval:
        test_net(
            'TEST',
            net,
            test_dataloader
            )

        brat_eval(
            'TEST',
            net,
            args
        )
    destroy_process_group()

def ddp_setup(rank: int, world_size: int, backend="nccl"):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend=backend, rank=rank, world_size=world_size)

def train_net(task_name, net, train_dataloader, valid_dataloader, test_dataloader, loss_fn, optimizer, args):

    if args.record_wandb and args.local_rank == 0:
        # use wandb to record weights
        wb_run = wandb.init(project='RE_scaling', name=args.record_wandb, entity="ucsd_sbks", config = args)
        wandb.watch(net, log='all', log_freq=500)
    # setup training
    print(f'Running train {task_name}')
    net.train()
    train_step_count = args.resume_from_step
    epoch_count = 0
    sample_counter = 0 # number of sample seen since last optimizer update
    early_stop = EarlyStopping(net, patience=6)

    # train loop
    for epoch_count in range(args.epochs):
        
        # early stop if patience number is reached
        if early_stop.early_stop:
            break
        train_dataloader.sampler.set_epoch(epoch_count)
        print(f'Begin epoch {epoch_count}')
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            # net.train_step(x, y, loss_fn, optimizer)
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
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
                    train_loss = test_net('TRAIN', net, train_dataloader, limit=(3000 // args.batch_size))
                    vali_loss = test_net('VALIDATION', net, valid_dataloader, limit=(.3*len(valid_dataloader.dataset)/args.batch_size))
                    early_stop(vali_loss, net, train_step_count)

                    if early_stop.early_stop:
                        break

                    if args.ckpt_dir is not None and args.local_rank == 0:
                        ckpt_path = os.path.join(args.ckpt_dir, f'{train_step_count}')
                        torch.save(net.state_dict(), ckpt_path)
                    net.train()
                    print(f'Resume epoch {epoch_count}')

                    if args.record_wandb and args.local_rank == 0:
                        wandb.log({'train loss': train_loss, 'vali loss': vali_loss}, step=train_step_count)

                if args.record_wandb and args.local_rank == 0:
                    wandb.log({'epoch': epoch_count}, step=train_step_count)
    # final test

    # get best model
    net, best_step_count = early_stop.get_best_model()

    # register activation hook if specified
    hooks = []
    for layer in args.record_activation:
        print(f'recording activation for layer: {layer}')
        hook = ActivationHook(layer)
        net.top_model.record_activation(layer, hook)
        hooks.append(hook)

    print('Train finished')
    test_net('TRAIN', net, train_dataloader)
    test_net('VALIDATION', net, valid_dataloader)
    test_net('TEST', net, test_dataloader)
    
    if args.ckpt_dir is not None and best_step_count > 0 and args.local_rank == 0:
        ckpt_path = os.path.join(args.ckpt_dir, f'best_model_{best_step_count}')
        torch.save(net.state_dict(), ckpt_path)

    # show recorded activations
    for hook in hooks:
        hook.vis_activation(args.ckpt_dir, to_wandb=args.record_wandb)

    if args.record_wandb and args.local_rank == 0:
        wb_run.finish()

def test_net(task_name, net, test_dataloader, limit=None):

    # setup testing
    print(f'Running test {task_name}')
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = len(cpr.get_label_map())
    confusion_mat = np.zeros((num_classes, num_classes))
    loss_lst = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(tqdm(test_dataloader)):
            # stop after a certain number of batch to save time
            if limit and i >= limit:
                break
            
            pred = net.predict(x_batch).cpu().numpy()
            y = y_batch.numpy()
            num_tested += len(y)
            num_correct += np.sum(pred == y)

            for p, y in zip(pred, y_batch):
                confusion_mat[y][p] += 1

            
            # calculate loss
            loss_fn = nn.CrossEntropyLoss()
            x = {k: v.cuda() for k, v in x_batch.items()}
            out = net(x)
            y = y_batch.cuda()
            loss = loss_fn(out, y)
            loss_lst.append(loss.item())

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
    print('Loss: ', np.mean(loss_lst))
    print('Confusion Matrix:')
    for row in confusion_mat:
        print('\t'.join([str(x) for x in row]))
    # print(confusion_mat)
    return np.mean(loss_lst)

def inference_net(task_name, net, args):

    # setup inference
    print(f'Running inference {task_name}')
    net.eval()

    vocab_filename = os.path.join(args.bert_state_path, 'vocab.txt')
    for pub_num in tqdm(os.listdir(args.inference_data)): # find all data folder in the dataset directory
        article_dir = os.path.join(args.inference_data, pub_num)
        assert os.path.isdir(article_dir)

        # one dataloader per acs files
        dataloader = get_acs_inference(
            data_filename=article_dir,
            vocab_filename=vocab_filename,
            max_seq_len=args.seq_len,
            batch_size=args.valid_batch_size
        )
        output = []
        with torch.no_grad():
            for _, (id1s, id2s, batch_x) in enumerate(dataloader):
                pred, score = net.predict(batch_x, return_score=True)
                score = score.cpu().tolist()
                pred = pred.cpu().tolist()
                for i in range(len(id1s)):
                    output.append((id1s[i], id2s[i], pred[i], score[i][pred[i]]))
        with open(os.path.join(article_dir, 're_output.tsv'), 'w', encoding="utf8") as fout:
            fout.write("id1\tid2\tclass\tconfidence\n")
            for _, (id1, id2, pred, score) in enumerate(output):
                fout.write(f"{id1}\t{id2}\t{pred}\t{score}\n")


def brat_eval(task_name, net, args):

    # set up
    print(f'Running BratEval {task_name}')
    net.eval()

    if task_name == 'TEST':
        vocab_filename = os.path.join(args.bert_state_path, 'vocab.txt')
        dataloader = get_brat_eval(
                data_filename=args.test_data,
                vocab_filename=vocab_filename,
                label_map = cpr.get_label_map(),
                max_seq_len=args.seq_len,
                batch_size=args.valid_batch_size
            )
        json_filename = 'data/merged/dev/merged.json'
        ann_folder = 'data/merged/brat_eval/dev_pred'

    elif task_name == 'VALIDATION':
        vocab_filename = os.path.join(args.bert_state_path, 'vocab.txt')
        dataloader = get_brat_eval(
                data_filename=args.valid_data,
                vocab_filename=vocab_filename,
                label_map = cpr.get_label_map(),
                max_seq_len=args.seq_len,
                batch_size=args.valid_batch_size
            )
        json_filename = 'data/merged/training_original/merged.json'
        ann_folder = 'data/merged/brat_eval/vali_pred'

    else:
        assert False, f"task name must be either 'TEST' or 'VALIDATION'"

    output = []
    idx_to_label = {v:k for k,v in cpr.get_label_map().items()}
    with torch.no_grad():
        for _, (x_batch, input_ids) in enumerate(tqdm(dataloader)):
            pred = net.predict(x_batch)
            pred = pred.cpu().tolist()
            for i in range(len(pred)):
                output.append((pred[i], input_ids[i]))

    # write results to file
    output_path = f'data/merged/brat_eval/re_{task_name}_output.tsv'
    with open(output_path, 'w', encoding="utf8") as fout:
        for _, (pred, input_id) in enumerate(output):
            fout.write(f"{input_id}\t{idx_to_label[pred]}\n")
    
    # transform the output to folder with .anns
    print('----Transform the output to folder with .anns----')
    convert_json_to_brat(json_filename, ann_folder)
    input_to_re(json_filename, output_path, ann_folder)


# for the reason of why we are not using type=bool in add_argument:
# see https://docs.python.org/3/library/argparse.html#type
# "
# The bool() function is not recommended as a type converter. 
# All it does is convert empty strings to False and non-empty strings to True. 
# This is usually not what is desired.
# "
def bool_string(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        assert False, f'invalid bool string {s}'

if __name__ == '__main__':

    args = parser.parse_args()
    # print out all package versions and name
    print('Installed packages:')
    print(os.system('pip list'))

    if args.device == 'gpu':
        device = torch.cuda.device(args.local_rank)
    #TODO: elif args.device == 'guadi':

    # calculate gradient accumulation parameter
    # if ga_batch_size is not specified, it will be set equal to batch_size
    # resulting in updates after every batch
    if args.ga_batch_size is None:
        args.ga_batch_size = args.batch_size

    # set valid batch size
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size

    # parse top hidden size
    args.top_hidden_size = [int(x) for x in args.top_hidden_size]

    # handle default checkpoint directory
    if args.ckpt_dir is None:
        args.ckpt_dir = 'weights/most_recent'
        print(f'WARNING: --ckpt-dir not provided, defaulting to {args.ckpt_dir}')
    if not os.path.exists(args.ckpt_dir):
        print(f'Creating ckpt_dir: {args.ckpt_dir}')
        os.makedirs(args.ckpt_dir)
    else:
        assert os.path.isdir(args.ckpt_dir), f'{args.ckpt_dir} has to be a directory'
        
    # set activation function for top model
    assert args.activation in ['ReLU', 'Tanh', 'GELU'], 'Activation function should be either ReLU, Tanh, or GELU.'

    # parse activation hook args
    args.record_activation = [int(x) for x in args.record_activation]

    print('Arguments:')
    print(args)

    print(f'CUDA availability: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU name: {torch.cuda.get_device_name(i)}')

    # call get label map once to set the default label map
    cpr.get_label_map(args.label_map_name)
    #TODO: Change this for gaudi
    world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
    mp.spawn(main, args=(world_size, args), nprocs=world_size)

