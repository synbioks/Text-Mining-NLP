import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.set_printoptions(suppress=True, linewidth=120)
import argparse
from tqdm import tqdm

from models.nets import get_end_to_end_net
from models.dataloader import get_train_valid
from models.dataloader import get_acs_inference
from models.optimizer import VarLROptim

from utils import cpr
from utils.activation_vis import ActivationHook

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
                    test_net('TRAIN', net, train_dataloader, limit=(3000 // args.batch_size))
                    test_net('VALIDATION', net, valid_dataloader, limit=(3000 // args.batch_size))
                    if args.ckpt_dir is not None:
                        ckpt_path = os.path.join(args.ckpt_dir, f'{train_step_count}')
                        torch.save(net.state_dict(), ckpt_path) 
                    net.train()
                    print(f'Resume epoch {epoch_count}')
    
    # final test

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
    
    if args.ckpt_dir is not None and train_step_count > 0:
        ckpt_path = os.path.join(args.ckpt_dir, f'{train_step_count}')
        torch.save(net.state_dict(), ckpt_path)

    # show recorded activations
    for hook in hooks:
        hook.vis_activation(args.ckpt_dir)

def test_net(task_name, net, test_dataloader, limit=None):

    # setup testing
    print(f'Running test {task_name}')
    net.eval()
    num_tested = 0
    num_correct = 0
    num_classes = len(cpr.get_label_map())
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
    for row in confusion_mat:
        print('\t'.join([str(x) for x in row]))
    # print(confusion_mat)

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
    parser.add_argument('--top-hidden-size', nargs='+', default=[1024, 1024])
    parser.add_argument('--freeze-bert', type=bool_string, default='False')
    parser.add_argument('--resume-from-ckpt', type=str)
    parser.add_argument('--resume-from-step', type=int, default=0)
    parser.add_argument('--train-data', type=str, default='data/merged/training/merged.txt')
    parser.add_argument('--valid-data', type=str, default='data/merged/dev/merged.txt')
    parser.add_argument('--label-map-name', type=str, default='merged')
    parser.add_argument('--inference-data', type=str, default='data/acs/acs-data')
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--balance-dataset', type=bool_string, default='False')
    parser.add_argument('--do-train', type=bool_string, default='True')
    parser.add_argument('--do-inference', type=bool_string, default='False')
    parser.add_argument('--activation', type=str, default='Tanh')
    parser.add_argument('--record-activation', nargs='+', default=[])
    args = parser.parse_args()

    # print out all package versions and name
    print('Installed packages:')
    print(os.system('pip list'))

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

    # initialize top model
    bert_hidden_size = 1024 if args.use_bert_large else 768
    out_size = len(cpr.get_label_map()) # calculate the output size
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
    # if validation data is not specified, we will do a train/valid 80-20 split on the train data
    train_dataloader, valid_dataloader = get_train_valid(
        args.train_data,
        args.valid_data,
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