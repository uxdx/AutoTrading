import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.linear import Linear

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, device):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size #hidden state
        self.output_size = output_size
        self.seq_length = seq_length # sequence length
        self.device = device

        self.layers = [
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128), #fully connected 1
            nn.ReLU(),
            nn.Linear(128, output_size), #fully connected last layer
            nn.ReLU(),
        ]
        self.loss_layer = nn.AdaptiveLogSoftmaxWithLoss(input_size, output_size,)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward()
        return x

    def forward(self,x,t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
