# author - Samantha Mahendran
# inspired by Lu, et al.-- https://github.com/Louis-udm/VGCN-BERT
import gc
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from torch.utils.data import DataLoader
from model_vgcn_bert import VGCN_Bert
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam  # , warmup_linear
from eval.write_predictions import Predictions

random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(44)
device = torch.device("cuda:0" if cuda_yes else "cpu")

class VGCN_BERT:
    def __init__(self, model, cleanData, buildGraph, initial_predictions=None, final_predictions=None, del_stop_words=False, model_type='VGCN_BERT', train_epochs=500,
                 dropout=0.2, batch_size=8, gcn_embedding_dim=16, learning_rate0= 3e-5, l2_decay=0.001):
        self.model = model
        self.data = cleanData
        self.graph = buildGraph

        MAX_SEQ_LENGTH = 200 + gcn_embedding_dim
        gradient_accumulation_steps = 1
        bert_model_scale = 'bert-base-uncased'
        do_lower_case = True
        warmup_proportion = 0.1
        perform_metrics_str = ['weighted avg', 'f1-score']

        cfg_vocab_adj = 'pmi'
        cfg_adj_npmi_threshold = 0.2
        cfg_adj_tf_threshold = 0
        classifier_act_func = nn.ReLU()

        resample_train_set = False  # if mse and resample, then do resample
        do_softmax_before_mse = True
        cfg_loss_criterion = 'cle'

        gcn_vocab_map = self.graph.vocab_map
        gcn_vocab = self.graph.vocab
        gcn_adj_list = self.graph.adj_list

        train_y = self.graph.train_y
        train_y_prob = self.graph.train_y_prob
        valid_y = self.graph.valid_y
        valid_y_prob = self.graph.valid_y_prob
        test_y = self.graph.test_y
        test_y_prob = self.graph.test_y_prob
        tfidf_X_list = self.graph.tfidf_X_list
        gcn_vocab_adj = self.graph.vocab_adj
        gcn_vocab_adj_tf = self.graph.vocab_adj_tf
        shuffled_clean_docs = self.graph.shuffled_clean_docs
        class_labels = self.graph.class_labels

        y = np.hstack((train_y, valid_y, test_y))
        y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

        examples = []
        for i, ts in enumerate(shuffled_clean_docs):
            ex = InputExample(i, ts.strip(), confidence=y_prob[i].astype(float), label=y[i])
            examples.append(ex)

        num_classes = len(self.graph.class_labels)
        gcn_vocab_size = len(gcn_vocab_map)
        train_size = len(train_y)
        valid_size = len(valid_y)
        test_size = len(test_y)

        indexs = np.arange(0, len(examples))
        train_examples = [examples[i] for i in indexs[:train_size]]
        valid_examples = [examples[i] for i in indexs[train_size:train_size + valid_size]]
        test_examples = [examples[i] for i in indexs[train_size + valid_size:train_size + valid_size + test_size]]

        if cfg_adj_tf_threshold > 0:
            gcn_vocab_adj_tf.data *= (gcn_vocab_adj_tf.data > cfg_adj_tf_threshold)
            gcn_vocab_adj_tf.eliminate_zeros()
        if cfg_adj_npmi_threshold > 0:
            gcn_vocab_adj.data *= (gcn_vocab_adj.data > cfg_adj_npmi_threshold)
            gcn_vocab_adj.eliminate_zeros()

        if cfg_vocab_adj == 'pmi':
            gcn_vocab_adj_list = [gcn_vocab_adj]
        elif cfg_vocab_adj == 'tf':
            gcn_vocab_adj_list = [gcn_vocab_adj_tf]
        elif cfg_vocab_adj == 'all':
            gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

        norm_gcn_vocab_adj_list = []
        for i in range(len(gcn_vocab_adj_list)):
            adj = gcn_vocab_adj_list[i]  # .tocsr() #(lr是用非norm时的1/10)
            print('  Zero ratio(?>66%%) for vocab adj %dth: %.8f' % (
                i, 100 * (1 - adj.count_nonzero() / (adj.shape[0] * adj.shape[1]))))
            adj = normalize_adj(adj)
            norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
        gcn_adj_list = norm_gcn_vocab_adj_list

        del gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_adj_list
        gc.collect()
        print(self.graph.class_labels)
        train_classes_num, train_classes_weight = get_class_count_and_weight(train_y, len(self.graph.class_labels))

        loss_weight = torch.tensor(train_classes_weight).to(device)

        tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

        def get_pytorch_dataloader(examples, tokenizer, batch_size, shuffle_choice, classes_weight=None,
                                   total_resample_size=-1):
            ds = CorpusDataset(examples, tokenizer, gcn_vocab_map, MAX_SEQ_LENGTH, gcn_embedding_dim)
            if shuffle_choice == 0:  # shuffle==False
                return DataLoader(dataset=ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  collate_fn=ds.pad)
            elif shuffle_choice == 1:  # shuffle==True
                return DataLoader(dataset=ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  collate_fn=ds.pad)
            elif shuffle_choice == 2:  # weighted resampled
                assert classes_weight is not None
                assert total_resample_size > 0
                weights = [classes_weight[0] if label == 0 else classes_weight[1] if label == 1 else classes_weight[2]
                           for _, _, _, _, label in dataset]
                sampler = WeightedRandomSampler(weights, num_samples=total_resample_size, replacement=True)
                return DataLoader(dataset=ds,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  num_workers=2,
                                  collate_fn=ds.pad)

        train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size, shuffle_choice=0)
        valid_dataloader = get_pytorch_dataloader(valid_examples, tokenizer, batch_size, shuffle_choice=0)
        test_dataloader = get_pytorch_dataloader(test_examples, tokenizer, batch_size, shuffle_choice=0)

        # total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
        total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * train_epochs)
        print('  Train_classes count:', train_classes_num)
        print('  Num examples for train =', len(train_examples), ', after weight sample:',
              len(train_dataloader) * batch_size)
        print("  Num examples for validate = %d" % len(valid_examples))
        print("  Batch size = %d" % batch_size)
        print("  Num steps = %d" % total_train_steps)

        # %%
        '''
        Train vgcn_bert model
        '''
        def predict(model, dataloader):
            # dataloader = get_pytorch_dataloader(examples, tokenizer, batch_size, shuffle_choice=0)
            predict_out = []
            confidence_out = []
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, _, label_ids, gcn_swop_eye = batch
                    score_out = model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
                    if cfg_loss_criterion == 'mse' and do_softmax_before_mse:
                        score_out = torch.nn.functional.softmax(score_out, dim=-1)
                    predict_out.extend(score_out.max(1)[1].tolist())
                    confidence_out.extend(score_out.max(1)[0].tolist())

            return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)

        def evaluate(model, gcn_adj_list, predict_dataloader, batch_size, epoch_th, dataset_name):
            # print("***** Running prediction *****")
            model.eval()
            predict_out = []
            all_label_ids = []
            ev_loss = 0
            total = 0
            correct = 0
            start = time.time()
            with torch.no_grad():
                for batch in predict_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
                    # the parameter label_ids is None, model return the prediction score
                    logits = model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

                    if cfg_loss_criterion == 'mse':
                        if do_softmax_before_mse:
                            logits = F.softmax(logits, -1)
                        loss = F.mse_loss(logits, y_prob)
                    else:
                        if loss_weight is None:
                            loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
                        else:
                            loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
                    ev_loss += loss.item()

                    _, predicted = torch.max(logits, -1)
                    predict_out.extend(predicted.tolist())
                    all_label_ids.extend(label_ids.tolist())
                    eval_accuracy = predicted.eq(label_ids).sum().item()
                    total += len(label_ids)
                    correct += eval_accuracy

                f1_metrics = f1_score(np.array(all_label_ids).reshape(-1),
                                      np.array(predict_out).reshape(-1), average='weighted')
                print("Report:\n" + classification_report(np.array(all_label_ids).reshape(-1),
                                                          np.array(predict_out).reshape(-1), digits=4))

            ev_acc = correct / total
            end = time.time()
            print('Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation'
                  % (epoch_th, ' '.join(perform_metrics_str), 100 * f1_metrics, 100. * ev_acc, dataset_name,
                     (end - start) / 60.0))
            print('--------------------------------------------------------------')
            return ev_loss, ev_acc, f1_metrics

        print("\n----- Running training -----")

        start_epoch = 0
        valid_acc_prev = 0
        perform_metrics_prev = 0

        model = VGCN_Bert.from_pretrained(bert_model_scale, gcn_adj_dim=gcn_vocab_size, gcn_adj_num=len(gcn_adj_list), gcn_embedding_dim=gcn_embedding_dim,
                                          num_labels=len(self.graph.class_labels))
        prev_save_step = -1

        model.to(device)
        optimizer = BertAdam(model.parameters(), lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps,
                             weight_decay=l2_decay)

        train_start = time.time()
        global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

        all_loss_list = {'train': [], 'valid': [], 'test': []}
        all_f1_list = {'train': [], 'valid': [], 'test': []}

        for epoch in range(start_epoch, train_epochs):
            print(epoch)
            tr_loss = 0
            ep_train_start = time.time()
            model.train()
            optimizer.zero_grad()
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                if prev_save_step > -1:
                    if step <= prev_save_step: continue
                if prev_save_step > -1:
                    prev_save_step = -1
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch

                logits = model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

                if cfg_loss_criterion == 'mse':
                    if do_softmax_before_mse:
                        logits = F.softmax(logits, -1)
                    loss = F.mse_loss(logits, y_prob)
                else:
                    if loss_weight is None:
                        loss = F.cross_entropy(logits, label_ids)
                    else:
                        loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, loss_weight.float())

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step_th += 1
                if step % 40 == 0:
                    print("Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(epoch, step, len(train_dataloader),cfg_loss_criterion,
                                                                                           loss.item(), (time.time() - train_start) / 60.0))

            print('--------------------------------------------------------------')
            valid_loss, valid_acc, perform_metrics = evaluate(model, gcn_adj_list, valid_dataloader, batch_size, epoch,'Valid_set')
            # test_loss, _, test_f1 = evaluate(model, gcn_adj_list, test_dataloader, batch_size, epoch, 'Test_set')
            all_loss_list['train'].append(tr_loss)
            all_loss_list['valid'].append(valid_loss)
            all_f1_list['valid'].append(perform_metrics)
            # all_loss_list['test'].append(test_loss)
            # all_f1_list['test'].append(test_f1)
            print("Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(epoch, tr_loss, valid_loss,
                                                                                            (time.time() - train_start) / 60.0))

        print('\n**Optimization Finished!,Total spend:', (time.time() - train_start) / 60.0)
        pred, confidence = predict(model, test_dataloader)

        Predictions(self.data, self.graph, pred, initial_predictions, final_predictions)
