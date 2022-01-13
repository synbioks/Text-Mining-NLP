from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class CLSTopModel(nn.Module):

    def __init__(self, bert_hidden_size, top_hidden_size, top_hidden_layers, out_size):
        super(CLSTopModel, self).__init__()

        dropout_p = 0.1

        # initialize hidden layers
        layers = []
        for i in range(top_hidden_layers):
            if i == 0:
                layers.extend([
                    nn.Dropout(p=dropout_p),
                    nn.Linear(bert_hidden_size, top_hidden_size),
                    nn.Tanh()
                ])
            else:
                layers.extend([
                    nn.Dropout(p=dropout_p),
                    nn.Linear(top_hidden_size, top_hidden_size),
                    nn.Tanh()
                ])
        
        # initialize output layers
        layers.extend([
            nn.Dropout(p=dropout_p),
            nn.Linear(top_hidden_size, out_size)
        ])

        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

class EndToEnd(nn.Module):

    def __init__(self, state_path, top_model):
        super(EndToEnd, self).__init__()
        self.bert = BertModel.from_pretrained(state_path)
        self.top_model = top_model

    def forward(self, x):
        y = self.bert(**x, return_dict=True).last_hidden_state
        y = y[:, 0, :]
        y = self.top_model(y)
        return y

    def train_step(self, x, y, loss_fn, optimizer):
        x = {k: v.cuda() for k, v in x.items()}
        y = y.cuda()
        optimizer.zero_grad()
        out = self.forward(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, x, return_score=False):
        x = {k: v.cuda() for k, v in x.items()}
        score = F.softmax(self.forward(x), dim=-1)
        pred = torch.argmax(score, dim=-1)
        if return_score:
            return pred, score
        else:
            return pred
    
    def bert_grad_required(self, required):
        for params in self.bert.parameters():
            params.requires_grad = required

def get_end_to_end_net(bert_weights_filename, bert_hidden_size, out_size, args):
    top_model = CLSTopModel(bert_hidden_size, args.top_hidden_size, args.top_hidden_layers, out_size)
    net = EndToEnd(bert_weights_filename, top_model)
    return net.cuda()