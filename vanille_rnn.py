################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        # weights and biases
        self.Whx = torch.nn.Parameter(torch.randn(217,num_hidden) / 1000)
        # self.Whx = torch.nn.Parameter(torch.randn(input_dim,num_hidden) / 1000)
        self.Whh = torch.nn.Parameter(torch.randn(num_hidden,num_hidden) / 1000)
        self.Wph = torch.nn.Parameter(torch.randn(num_hidden, num_classes) / 1000)
        self.bh = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bp = torch.nn.Parameter(torch.zeros(1, num_classes))

        # number of timesteps
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(self.batch_size, self.num_hidden)
        # print()
        # print('h',h.shape)
        for t in range(140):

            # print(x[:,t:(t+1)*217].squeeze(dim=1).shape,self.Whx.shape)
            # print(x[:,t,:].shape, self.Whx.shape)

            # W1 = x[:,t].unsqueeze(dim=1) @ self.Whx
            W1 = x[:,t,:] @ self.Whx
            W2 = h @ self.Whh

            # print('w1,w2',W1.shape, W2.shape)
            # print('h,whh',h.shape,self.Whh.shape)

            h = torch.tanh(W1 + W2 + self.bh)
            # print(h.shape,W1.shape,W2.shape)

        p = h @ self.Wph + self.bp
        # print('h2',h.shape)
        # print('Wph',self.Wph.shape)
        # print('bp',self.bp.shape)

        # print('HWPH',(h @ self.Wph).shape)

        # print('p',p.shape)
        return p, None
