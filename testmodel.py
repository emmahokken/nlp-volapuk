import torch
import torch.nn as nn

import numpy as np

class modelRNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(modelRNN, self).__init__()

        self.input_size = input_size

        self.rnn = nn.RNN(input_size=input_size,
                            hidden_size=10,
                            num_layers=2,
                            bias=True,
                            nonlinearity='relu',
                            bidirectional=False,
                            batch_first=True)

        self.lin = nn.Linear(10, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.softmax(self.lin(out))

        return out, hidden
