import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class CLSTopModel(nn.Module):

    def __init__(self):
        super(CLSTopModel, self).__init__()

        input_size = 1024
        hidden_size = 1024
        dropout_p = 0.1
        num_class = 14

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, num_class)
        )
    
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

def get_end_to_end_net(bert_weights_filename):
    top_model = CLSTopModel()
    net = EndToEnd(bert_weights_filename, top_model)
    return net.cuda()