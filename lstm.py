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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.Wgx = torch.nn.Parameter(torch.randn(input_dim,num_hidden) / 1000)
        self.Wgh = torch.nn.Parameter(torch.randn(num_hidden,num_hidden) / 1000)

        self.Wix = torch.nn.Parameter(torch.randn(input_dim, num_hidden) / 1000)
        self.Wih = torch.nn.Parameter(torch.randn(num_hidden,num_hidden) / 1000)

        self.Wfx = torch.nn.Parameter(torch.randn(input_dim,num_hidden) / 1000)
        self.Wfh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden) / 1000)

        self.Wox = torch.nn.Parameter(torch.randn(input_dim,num_hidden) / 1000)
        self.Woh = torch.nn.Parameter(torch.randn(num_hidden,num_hidden) / 1000)

        self.Wph = torch.nn.Parameter(torch.randn(num_hidden,num_classes) / 1000)

        self.bg = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bi = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bf = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bo = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bp = torch.nn.Parameter(torch.zeros(1, num_classes))

        # number of timesteps
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

    def forward(self, x):
        # Implementation here ...

        h = torch.zeros(self.batch_size, self.num_hidden)
        c = torch.zeros(self.batch_size, self.num_hidden)
        for t in range(self.seq_length):

            xt = x[:,t].unsqueeze(dim=1).float()

            # values dependent on h^{t-1}
            # multiplications have been inversed (x @ W instead of W @ x)
            # due to reversed initialization
            g = torch.tanh(xt @ self.Wgx + h @ self.Wgh + self.bg)
            i = torch.sigmoid(xt @ self.Wix + h @ self.Wih + self.bi)
            f = torch.sigmoid(xt @ self.Wfx + h @ self.Wfh + self.bf)
            o = torch.sigmoid(xt @ self.Wox + h @ self.Woh + self.bo)

            # dependent on c^{t-1}
            c = g * i + c * f

            # update h
            h = torch.tanh(c) * o
        p = h @ self.Wph + self.bp

        return p