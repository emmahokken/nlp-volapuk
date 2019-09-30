import torch
import torch.nn as nn

import numpy as np

class modelRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, device='cpu'):
        super(modelRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity='tanh',
                            bias=True,
                            batch_first=True)

        # self.lstm = nn.LSTM(input_size=input_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=num_layers,
        #                     bidirectional=False,
        #                     batch_first=True)

        self.lin = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()

        # print("inside model init\n")
        # print('num_layers:', num_layers)
        # print('input_size:', input_size)
        # print('hidden_size:', hidden_size)


    def forward(self, x):
        # print('x size', x.size())

        # have to do something with hidden?
        out, hidden = self.rnn(x)
        out = self.softmax(self.lin(out))

        return out, hidden
