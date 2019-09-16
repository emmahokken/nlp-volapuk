import torch
import torch.nn as nn

import numpy as np

class modelRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, device='cpu'):
        super(modelRNN, self)__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity='tanh',
                            bias=True,
                            batch_first=False,
                            device=device)

    def forward(self, x):
        # have to do something with hidden?
        out = self.rnn(x)

        return out
