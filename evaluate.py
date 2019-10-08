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
import operator
import baseline

def evaluate(args):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    data = DataSet()

    # vocab_size = 217
    # args.embedding_size = 128
    # args.classes = 146
    # args.rnn = 'LSTM'
    # args.mean_seq = False
    # args.hidden_size = 128
    # args.layers = 2

    args.vocab_size = len(data.char2int)
    args.embedding_size = 256#len(data.char2int)#128
    args.classes = len(data.languages)
    args.num_hidden = 64
    model = VolapukModel(vocab_size=args.vocab_size, embed_size=args.embedding_size, num_output=args.classes,
            hidden_size=args.num_hidden, num_layers=args.num_layers, batch_first=True, k=args.k).to(device=device)

    # model = VolapukModel(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes,
    #         hidden_size=args.hidden_size, num_layers=args.layers, batch_first=True).to(device=device)

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

        out, _, _ = model.forward(X, (torch.ones(args.batch_size)*args.batch_size).long())
        acc, correct_dict, total_dict = accuracy(out, Y, correct_dict, total_dict)

    print(correct_dict)

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
        acc_per_lan[lan2language[lan]] = (total_dict[data.lan2int[lan]] / correct_dict[data.lan2int[lan]]).mean().item()

    print(acc_per_lan)

    # baseline_acc_per_lan = baseline.unigram_baseline(args)

    # barplot_languages(acc_per_lan, baseline_acc_per_lan)
    # barplot_languages(acc_per_lan, baseline_acc_per_lan, acc_per_lan)
    # plot_languages(acc_per_lan)

    return acc_per_lan#, baseline_acc_per_lan

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

def barplot_languages(acc_per_lan, baseline_acc_per_lan, lambda_acc_per_lan):

    keys = [key for key, val in reversed(sorted(acc_per_lan.items(), key=operator.itemgetter(1)))]
    vals = [val for key, val in reversed(sorted(acc_per_lan.items(), key=operator.itemgetter(1)))]
    lambda_vals = [lambda_acc_per_lan[key] for key in keys]
    base_vals = [baseline_acc_per_lan[key] for key in keys]

    print(len(keys))


    # for n_keys, n_vals, n_base_vals in zip([keys[:50], keys[50:100], keys[100:]], [vals[:50], vals[50:100], vals[100:]], [base_vals[:50], base_vals[50:100], base_vals[100:]]):
    for n_keys, n_vals, n_base_vals, n_lambda_vals in zip([keys[:78], keys[78:]], [vals[:78], vals[78:]], [base_vals[:78], base_vals[78:]], [lambda_vals[:78], lambda_vals[78:]]):

        y_pos = np.arange(len(n_keys))
        ax = plt.subplot(111)
        w = 0.3

        # plt.bar(y_pos, vals, align='center', alpha=0.5)
        # plt.xticks(y_pos, keys, rotation='vertical', fontsize=10)

        ax.plot(y_pos, n_vals, color='b',label='LSTM-model')
        # ax.bar(y_pos-w, n_vals, width=w, color='b', align='center',label='LSTM-model')
        ax.bar(y_pos, n_lambda_vals, width=w, color='g', align='center',label='\u03BB-LSTM-model')
        ax.bar(y_pos+w, n_base_vals, width=w, color='r', align='center',label='BoC')
        plt.xticks(y_pos, n_keys, rotation='vertical', fontsize=7)
        ax.xaxis_date()
        ax.autoscale(tight=True)
        ax.legend()

        plt.show()

    y_pos = np.arange(len(keys))
    plt.grid(b=True, axis='x')
    plt.scatter(y_pos, vals, color='b',label='LSTM-model')
    plt.scatter(y_pos, lambda_vals, color='g',label='\u03BB-LSTM-model')
    plt.scatter(y_pos, base_vals, color='r',label='BoC')
    plt.xticks(y_pos, keys, rotation='vertical', fontsize=7)
    plt.show()

def save_to_csv(acc_per_lan, baseline_acc_per_lan, lambda_acc_per_lan):
    keys = [key for key, val in reversed(sorted(acc_per_lan.items(), key=operator.itemgetter(1)))]
    vals = [val for key, val in reversed(sorted(acc_per_lan.items(), key=operator.itemgetter(1)))]
    lambda_vals = [lambda_acc_per_lan[key] for key in keys]
    base_vals = [baseline_acc_per_lan[key] for key in keys]

    with open('evaluation_data.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['LSTM-model', '\u03BB-LSTM-model', 'BoC'])
        for i, key in enumerate(keys):
            writer.writerow([keys[i], vals[i], lambda_vals[i], base_vals[i]])

def print_csv():
    big_table = []

    boc = []
    lstm = []
    lambd = []
    with open('evaluation_data.csv', mode='r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for ind, row in enumerate(csv_reader):
            try:
                boc.append(float(row[-1]))
                lambd.append(float(row[-2]))
                lstm.append(float(row[-3]))
            except:
                pass
            # line = ''
            for i, r in enumerate(row):
                try:
                    row[i] = str(round(float(r),3))
                except:
                    row[i] = r
            # print(ind, ' & '.join(row), '\\\\')
            big_table.append(' & '.join(row))
    # print(len(big_table[1:74]),len(big_table[74:]))
    # for b1, b2 in zip(big_table[1:74], big_table[74:]):
    #     print(b1, '&', b2, '\\\\')
    big1 = big_table[1:50]
    big2 = big_table[50:99]
    big3 = big_table[99:]
    big3.append(f'Total & {np.mean(lstm[1:])} & {np.mean(lambd[1:])} & {np.mean(boc[1:])}')
    print(big3)
    # print(len(big1), len(big2), len(big3))
    for b1, b2, b3 in zip(big1, big2, big3):
        print(b1, '&', b2, '&', b3, '\\\\')
    # print(len(big_table[1:50]),len(big_table[50:99]),len(big_table[99:]))
    # for b1, b2, b3 in zip(big_table[1:49], big_table[49:49*2], big_table[49*2:]):
    #     print(b1, '&', b2, '&', b3, '\\\\')
    # print(np.argmax(boc[1:]),max(boc[1:]))
    # print('boc  ', np.mean(boc[1:]))
    # print('lambd', np.mean(lambd[1:]))
    # print('lstm ', np.mean(lstm[1:]))



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
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--PATH', type=str, default="models/model", help="Model name to save")
    parser.add_argument('--load_model', type=bool, default=True, help="Load model from PATH")

    parser.add_argument('--seed', type=int, default=42, help="Set seed")
    parser.add_argument('--evaluate_steps', type=int, default=25, help="Evaluate model every so many steps")


    parser.add_argument('--k', type=int, default=140, help="Num of characters for prediction")

    args = parser.parse_args()

    print_csv()

    # Train the model
    # retrain(args)

    # args.PATH should be the no lambda model
    acc_per_lan = evaluate(args)

    args.PATH = 'models/model__b128_h128_l2_s42_it20000_k35_Mon_Oct_7_11:30:12_2019' # this is lambda model
    lambda_acc_per_lan = evaluate(args)

    baseline_acc_per_lan = baseline.unigram_baseline(args)

    barplot_languages(acc_per_lan, baseline_acc_per_lan, lambda_acc_per_lan)

    save_to_csv(acc_per_lan, baseline_acc_per_lan, lambda_acc_per_lan)
