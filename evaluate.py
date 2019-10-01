from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
from model import modelRNN
import torch.nn as nn
import torch.optim as optim
from volapukModel import VolapukModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def evaluate(args):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    data = DataSet()

    vocab_size = 217
    args.embedding_size = 128
    args.classes = 146
    args.rnn = 'LSTM'
    args.mean_seq = False
    args.hidden_size = 128
    args.layers = 2

    model = VolapukModel(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes,
            hidden_size=args.hidden_size, num_layers=args.layers, batch_first=True).to(device=device)

    if args.load_model:
        print(f'Load model {args.PATH}.p')
        model.load_state_dict(torch.load(f'{args.PATH}.p', map_location=device))

    correct_dict = {data.lan2int[lan]: torch.zeros(len(data.languages)) for lan in data.languages}
    total_dict = {data.lan2int[lan]: torch.zeros(len(data.languages)) for lan in data.languages}

    loss = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    temp_batch_size = args.batch_size

    losses = []
    accuracies = []
    steps = []

    for i in tqdm(range(args.training_steps)):

        # Get batch and targets, however not in correct format
        # batch, targets = data.get_next_batch(args.batch_size)
        batch, targets = data.get_next_test_batch(args.batch_size)
        list_of_one_hot_X = []

        # Convert batch to X
        y = []
        languages = list(data.languages)
        # print(data.languages)
        for par, target in zip(batch, targets):
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

            list_of_one_hot_X.append(x_tensor)
            y.append(languages.index(target))

        X = torch.stack(list_of_one_hot_X,dim=0).squeeze()
        Y = torch.tensor(y).type(torch.LongTensor).view(-1, 1).squeeze()

        args.batch_size = Y.shape[0]

        X = X.to(device)
        Y = Y.to(device)

        out = model.forward(X, (torch.ones(args.batch_size)*args.batch_size).long())
        acc, correct_dict, total_dict = accuracy(out, Y, correct_dict, total_dict)

    acc_per_lan = {}
    for lan in data.languages:
        acc_per_lan[lan] = (total_dict[data.lan2int[lan]] / correct_dict[data.lan2int[lan]]).mean()

    print(acc_per_lan)
    print(acc)

    for key in acc_per_lan:
        print(key, acc_per_lan[key].item())

def accuracy(predictions, targets, correct_dict, total_dict):
    _, ind = torch.max(predictions, dim=1)
    acc = (ind == targets).float().mean()

    for i, t in zip(ind, targets):
        correct_dict[i.item()] += 1
        if i == t:
            total_dict[i.item()] += 1


    return acc, correct_dict, total_dict

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--PATH', type=str, default="models/model", help="Model name to save")
    parser.add_argument('--load_model', type=bool, default=True, help="Load model from PATH")

    parser.add_argument('--seed', type=int, default=42, help="Set seed")
    parser.add_argument('--evaluate_steps', type=int, default=25, help="Evaluate model every so many steps")

    args = parser.parse_args()

    # Train the model
    # retrain(args)
    evaluate(args)
