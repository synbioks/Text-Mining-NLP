import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BiLSTMTopModel(nn.Module):

    def __init__(self, lstm_hidden_size=512, num_class=6, dropout_p=0.1):
        super(BiLSTMTopModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout_p
        )
        self.dropout_lstm = nn.Dropout(p=dropout_p)
        self.dropout_linear = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(lstm_hidden_size * 2, num_class)

    def forward(self, x):
        y = x.last_hidden_state
        y = self.dropout_lstm(y)
        y = self.lstm(y)[0][:, -1, :]

        y = self.dropout_linear(y)
        y = self.linear(y)
        return y

class FirstTokenPoolingTopModel(nn.Module):

    def __init__(self, num_class=6, dropout_p=0.1):
        super(FirstTokenPoolingTopModel, self).__init__()
        self.linear = nn.Linear(768, 768)
        self.linear_act = nn.Tanh()
        self.linear2_dropout = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(768, num_class)

    def forward(self, x):

        # get the hidden representation of the first token
        y = x.last_hidden_state
        y = self.linear_act(self.linear(y[:, 0, :]))
        y = self.linear2(self.linear2_dropout(y))
        return y

class HiddenPoolingTopModel(nn.Module):

    def __init__(self, num_class=6, dropout_p=0.1, last_x=4):
        super(HiddenPoolingTopModel, self).__init__()
        self.last_x = last_x
        self.linear = nn.Linear(768 * self.last_x, 1024)
        self.linear_act = nn.Tanh()
        self.linear2_dropout = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(1024, num_class)

    def forward(self, x):
        last_hidden = x.hidden_states[-self.last_x:]
        y = []
        for item in last_hidden:
            y.append(item[:, 0, :])
        y = torch.cat(y, dim=-1)
        y = self.linear_act(self.linear(y))
        y = self.linear2(self.linear2_dropout(y))
        return y
        

class BertRE(nn.Module):

    def __init__(self, state_path, top_model, is_train=False):
        super(BertRE, self).__init__()
        self.bert = BertModel.from_pretrained(state_path)
        self.top_model = top_model

    def forward(self, x):
        y = self.bert(**x, return_dict=True, output_hidden_states=True)
        y = self.top_model(y)
        return y

if __name__ == "__main__":

    net = BertRE("../../weights/torch/", BiLSTMTopModel())
    net = net.cuda()

    tokenizer = BertTokenizer("../../weights/torch/vocab.txt", do_lower_case=False)
    sample = tokenizer("Hi there", return_tensors="pt")

    def map_to_cuda(x):
        res = {}
        for k, v in x.items():
            res[k] = v.cuda()
        return res

    print(net(map_to_cuda(sample)))
