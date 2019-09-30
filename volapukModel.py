import torch
import torch.nn as nn

class VolapukModel(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VolapukModel, self).__init__()
        lstm = nn.LSTM(input_dim, 256, num_hidden)
        l1 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = l1(lstm(x))
        return out