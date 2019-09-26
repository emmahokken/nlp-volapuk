from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
from model import modelRNN
import torch.nn as nn



def train(args):

    data = DataSet()
    # parse_chars(data)

    # HARDCODED WATCH THIS SPACE
    model = modelRNN(217, args.num_hidden, args.num_layers)
    loss = nn.CrossEntropyLoss()


    for i in range(args.training_steps):

        # get batch and targets, however not in correct format
        batch, targets = data.get_next_batch(args.batch_size)
        list_of_one_hot_X = []

        # Convert batch to X
        for par in batch:
            # We cutoff to only use the final 140 characters at the end.
            # This is done, as the first will have a lot of meaningless information, such as pronunciaton
            # This way it is easier for the batches to be read, as well as the bias for the length of the text to be removed.
            par = par[-140:]
            x = []
            for char in par:
                try:
                    x.append(data.char2int[char])
                except:
                    x.append(data.char2int['☃'])
            x_tensor = torch.tensor(x).type(torch.LongTensor).view(-1, 1)
            x_one_hot = torch.zeros(x_tensor.size()[0], len(data.char2int)).scatter_(1, x_tensor, 1)
            list_of_one_hot_X.append(x_one_hot)

        X = torch.stack(list_of_one_hot_X)

        # convert targets to Y
        languages = list(data.languages)
        languages.sort()
        y = []
        for target in targets:
            y.append(languages.index(target))
        y_tensor = torch.tensor(y).type(torch.LongTensor).view(-1, 1)
        Y = torch.zeros(y_tensor.size()[0], len(data.languages)).scatter_(1, y_tensor, 1)


        print('X', X.shape)
        print('Y', Y.shape)
        # print(Y.shape)


        out, hidden = model.forward(X)

        # print('\n\n',out.shape)
        # print(Y.shape, '\n\n')
        #
        output = loss(out, Y.long())

        output.backward()

        if i % 100 == 0:
            print(output.item())
            acc = accuracy(out.detach().numpy(), Y)



def accuracy(predictions, targets):
    pred = torch.argmax(predictions, axis=1)
    tar = torch.argmax(targets, axis=1)

    return (pred - tar).mean()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=128, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    # Train the model
    train(args)
