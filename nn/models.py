import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import dropout
from torch.nn.modules.linear import Linear

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, feature_length):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size #hidden state
        self.feature_length = feature_length # sequence length

        self.layers = [
            nn.LSTM(input_size,hidden_size,2, dropout=0.2,batch_first=True)
        ]
        self.loss_layer = nn.AdaptiveLogSoftmaxWithLoss(input_size, output_size,)

    def predict(self, x):
        # input shape: (batch_size, feature_length, input_size) e.g. (10, 2, 26)
        for layer in self.layers:
            x = layer.forward()
        return x

    def forward(self,x,t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
rnn_type='LSTM'
lstm = getattr(nn, rnn_type)(60, 20, 1, batch_first=True, dropout=0.2, bidirectional=1)
class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)


    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        return rnn_output