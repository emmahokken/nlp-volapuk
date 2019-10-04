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

import csv

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

    model.eval()

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

    lan2language = {}
    with open('../data/wili-2018/labels.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if row[1] == 'Swahili (macrolanguage)':
                    lan2language[row[0]] = 'Swahili'
                    continue
                lan2language[row[0]] = row[1]

    acc_per_lan = {}
    for lan in data.languages:
        acc_per_lan[lan2language[lan]] = (total_dict[data.lan2int[lan]] / correct_dict[data.lan2int[lan]]).mean()

    plot_languages(acc_per_lan)

def plot_languages(acc_per_lan):
    tenners = []
    twenties = []
    thirties = []
    fourties = []
    fifties = []
    sixties = []
    seventies = []
    eighties = []
    nineties = []
    counts = []
    for i, key in enumerate(acc_per_lan):
        counts.append(i)
        if acc_per_lan[key] > 0.90:
            nineties.append(key)
        elif acc_per_lan[key] > 0.80:
            eighties.append(key)
        elif acc_per_lan[key] > 0.70:
            seventies.append(key)
        elif acc_per_lan[key] > 0.60:
            sixties.append(key)
        elif acc_per_lan[key] > 0.50:
            fifties.append(key)
        elif acc_per_lan[key] > 0.40:
            fourties.append(key)
        elif acc_per_lan[key] > 0.30:
            thirties.append(key)
        elif acc_per_lan[key] > 0.20:
            twenties.append(key)
        elif acc_per_lan[key] > 0.10:
            tenners.append(key)

    max = 10000
    min = 50

    for i, lan in enumerate(nineties):
        lin = np.linspace(min, max * 0.95, len(nineties))
        plt.text(.902, lin[i], lan)
    for i, lan in enumerate(eighties):
        lin = np.linspace(min, max * 0.8, len(eighties))
        plt.text(.802, lin[i], lan)
    for i, lan in enumerate(seventies):
        lin = np.linspace(min, max * 0.7, len(seventies))
        plt.text(.702, lin[i], lan)
    for i, lan in enumerate(sixties):
        lin = np.linspace(min, max * 0.6, len(sixties))
        plt.text(.602, lin[i], lan)
    for i, lan in enumerate(fifties):
        lin = np.linspace(min, max * 0.5, len(fifties))
        plt.text(.502, lin[i], lan)
    for i, lan in enumerate(fourties):
        lin = np.linspace(min, max * 0.4, len(fourties))
        plt.text(.402, lin[i], lan)
    for i, lan in enumerate(thirties):
        lin = np.linspace(min, 1500, len(thirties))
        plt.text(.302, lin[i], lan)
    # for i, lan in enumerate(twenties):
    #     lin = np.linspace(min, 1500, len(twenties))
    #     plt.text(.202, lin[i], lan)
    # for i, lan in enumerate(tenners):
    #     lin = np.linspace(min, 1500, len(tenners))
    #     plt.text(.102, lin[i], lan)

    filler = np.arange(0.3,1.1,0.1)

    for k in filler:
        plt.plot([k for i in range(max)], [i for i in range(max)], color='black')

    colours = ['rebeccapurple', 'orange', 'blue', 'red', 'cyan', 'yellow', 'green', 'magenta', 'dodgerblue']

    for j, k in enumerate(filler):
        try:
            plt.fill_between([k, filler[j + 1]], 0, max, alpha=0.2, color=colours[j])
        except:
            pass

    plt.ylim(0,max)
    plt.yticks([])

    plt.xlabel("Accuracy")
    plt.title("Languages in accuracy bins")
    plt.savefig('acc_per_lan')
    plt.show()


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
