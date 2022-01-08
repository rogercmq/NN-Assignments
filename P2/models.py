import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LinearPredictor(nn.Module):
    def __init__(self, seq_length, hidden_size=32, input_size=1, num_classes=56):
        """
            seq_length:  输入长度
            hidden_size: 线性层宽度
            input_size: 数据维度(=1 因为只有一个浮点数)
            num_classes: 输入长度(=56 多步预测)
        """
        super(LinearPredictor, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.fc1 = nn.Linear(seq_length*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):      
        out = x.view(-1, self.seq_length*self.input_size)     
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


class GRUPredictor(nn.Module):
    def __init__(self, seq_length, num_layers, hidden_size=32, input_size=1, num_classes=56):
        super(GRUPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))    
        _, h_out  = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = F.relu(self.fc1(h_out))
        out = self.fc2(out)
        return out


class LSTMPredictor(nn.Module):
    def __init__(self, seq_length, num_layers, hidden_size=32, input_size=1, num_classes=56):
        super(LSTMPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out
