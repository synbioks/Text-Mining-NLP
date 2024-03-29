import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import apex

from transformers import BertModel

class CLSTopModel(nn.Module):

    def __init__(self, name="CHEMPROT", layers=3, use_loc_ids=False, input_size=768):
        super(CLSTopModel, self).__init__()
        # input_size = 768
        hidden_size = 1024
        assert(name in ["CHEMPROT", "DRUGPROT"])
        
        if use_loc_ids:
            input_size *= 3
        
        if name == "CHEMPROT":
            num_class = 4
        elif name == "DRUGPROT":
            num_class = 14
        dropout_p = 0.1
        
        layers_list = []
        if layers == 3:
            layers_list = [
                nn.Dropout(p=dropout_p),
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_size, num_class)]
        elif layers == 1:
            layers_list = [
                nn.Dropout(p=dropout_p),
                nn.Linear(input_size, num_class)]
        else:
            raise NotImplementedError("number of latyers in top level model can only be one or three.")

        self.fc = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.fc(x)

class TopLSTM(nn.Module):

    def __init__(self):
        super(TopLSTM, self).__init__()
        input_size = 768
        hidden_size = 128
        num_layers =  1
        num_class = 4
        dropout_p = 0.1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_class)
        )
    
    def forward(self, x):
        batch_size = x.size()[0]
        output, (hidden, cell) = self.rnn(x)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)[-1]
        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)
        return self.fc(hidden)

    # def train_step(self, x, y, loss_fn, optimizer):
    #     x = x.cuda()
    #     y = y.cuda()
    #     optimizer.zero_grad()
    #     out = self.forward(x[:, 0, :]) # take the CLS token
    #     loss = loss_fn(out, y)
    #     loss.backward()
    #     optimizer.step()
    #     return loss

    # def predict(self, x, return_score=False):
    #     x = x.cuda()
    #     score = F.softmax(self.forward(x[:, 0, :]), dim=-1)
    #     pred = torch.argmax(score, dim=-1)
    #     if return_score:
    #         return pred, score
    #     else:
    #         return pred

class EndToEnd(nn.Module):

    def __init__(self, state_path, top_model=CLSTopModel(), use_loc_ids=False):
        super(EndToEnd, self).__init__()
        self.bert = BertModel.from_pretrained(state_path)
        self.top_model = top_model
        self.clip_param_grad = None
        self.use_loc_ids = use_loc_ids

    def forward(self, x, loc_ids):
        if self.use_loc_ids:
            loc_ids = loc_ids.repeat(1, 1, 768)
            y = self.bert(**x, return_dict=True).last_hidden_state
            y = torch.gather(y, 1, loc_ids).reshape(y.shape[0], -1)
        else:
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

    def dep_train_step_new(self, x, y, loc_ids, loss_fn, optimizer, scheduler):
        x = {k: v.cuda() for k, v in x.items()}; y = y.cuda(); loc_ids = loc_ids.cuda()
        optimizer.zero_grad()
        out = self.forward(x, loc_ids)
        loss = loss_fn(out, y)
        loss.backward()
        if self.clip_param_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_param_grad)
        optimizer.step()
        scheduler.step()
        return loss

    def train_step_new(self, x, y, loc_ids, loss_fn, divider_loss=1):
        x = {k: v.cuda() for k, v in x.items()}; y = y.cuda(); loc_ids = loc_ids.cuda()
        out = self.forward(x, loc_ids)
        loss = loss_fn(out, y)/divider_loss
        loss.backward()
        if self.clip_param_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_param_grad)
        return loss

    def predict(self, x, loc_ids, return_score=False):
        x = {k: v.cuda() for k, v in x.items()}; loc_ids = loc_ids.cuda()
        score = F.softmax(self.forward(x, loc_ids), dim=-1)
        pred = torch.argmax(score, dim=-1)
        if return_score:
            return pred, score
        else:
            return pred
