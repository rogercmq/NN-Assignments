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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() 
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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


class Encoder(nn.Module):
    def __init__(self, hid_dim=64, n_layers=2, dropout=0.2, input_dim=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.network = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.network(x)        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, hid_dim=64, n_layers=2, dropout=0.2, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.network = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.network(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        assert self.encoder.hid_dim == self.decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, "Encoder and decoder must have equal number of layers!"
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        hidden, cell = self.encoder(x)
        x_ = torch.zeros_like(x)
        output, hidden, cell = self.decoder(x_, hidden, cell)
        output = output.squeeze(-1)
        return output
