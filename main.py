from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from volapukModel import VolapukModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from termcolor import colored
import csv

import os

import evaluate

def get_X_Y_from_batch(batch, targets, data, device):
    batch_x = []
    batch_y = []

    # Convert batch to X
    languages = list(data.languages)
    for par, target in zip(batch, targets):
        # We cutoff to only use the final 140 characters at the end.
        # This is done, as the first will have a lot of meaningless information, such as pronunciaton
        # This way it is easier for the batches to be read, as well as the bias for the length of the text to be removed.
        par = par[-140:]
        x = []
        # For all characters in the paragraph, try to append it, else it should be _unk_
        for char in par:
            try:
                x.append(data.char2int[char])
            except:
                x.append(data.char2int['☃'])
        x_tensor = torch.tensor(x).type(torch.LongTensor).view(-1, 1)
        batch_x.append(x_tensor)
        batch_y.append(languages.index(target))

    X = torch.stack(batch_x,dim=0).squeeze().to(device)
    Y = torch.tensor(batch_y).type(torch.LongTensor).view(-1, 1).squeeze().to(device)

    return X,Y

def train(args):
    # Variational Observed LAnguage Predictor
    print('#################################')
    print('############ Volapuk ############')
    print('#################################\n')
    # print('#################################')
    # print('########## Variational ##########')
    # print('########## Observed    ##########')
    # print('########## Language    ##########')
    # print('########## Predictor   ##########')
    # print('########## Using       ##########')
    # print('########## K           ##########')
    # print('#################################\n')

    params_id = '_b'+str(args.batch_size)+'_h'+str(args.num_hidden)+'_l'+str(args.num_layers)+'_s'+str(args.seed)+'_it'+str(args.training_steps)

    timestamp = time.asctime(time.localtime(time.time())).replace('  ',' ').replace(' ','_')

    modelname_id = params_id+'_'+timestamp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using available device {device}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = DataSet()

    args.vocab_size = len(data.char2int)
    args.classes = len(data.languages)

    print(args)

    model = VolapukModel(vocab_size=args.vocab_size, embed_size=args.embedding_size, num_output=args.classes,
            hidden_size=args.num_hidden, num_layers=args.num_layers, batch_first=True, importance_sampler=args.importance_sampler).to(device=device)

    if args.load_PATH is not None:
        print(f'Load model {args.load_PATH}')
        model.load_state_dict(torch.load(f'{args.load_PATH}'))


    print('\n#################################')
    print('############# Model #############')
    print('#################################')
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    temp_batch_size = args.batch_size

    losses = []
    accuracies = []
    steps = []

    print('\n###################################')
    print('############ Languages ############')
    print('###################################')
    print(data.languages)


    if not os.path.exists('csv'):
        os.makedirs('csv')

    for i in tqdm(range(args.training_steps)):

        # Get batch and targets, however not in correct format
        batch, targets = data.get_next_batch(args.batch_size)
        X,Y = get_X_Y_from_batch(batch, targets, data, device)

        lambd = 0.1
        out, _, mask_loss = model.forward(X, (torch.ones(args.batch_size)*args.batch_size).long()) # lstm from dl

        optimizer.zero_grad()
        loss = criterion(out, Y)
        loss = loss + lambd * mask_loss/args.batch_size
        loss.backward()
        optimizer.step()

        if i % args.evaluate_steps == 0:
            # Get batch from test data
            model.eval()
            test_batch, test_targets = data.get_next_test_batch(args.batch_size)
            test_X, test_Y = get_X_Y_from_batch(test_batch, test_targets, data, device)
            test_out, test_mask, test_mask_loss = model.forward(test_X, (torch.ones(args.batch_size)*args.batch_size).long())
            test_loss = criterion(test_out, test_Y)
            test_loss = test_loss + lambd * test_mask_loss/args.batch_size

            # Get max indices for accuracy
            _, ind = torch.max(test_out, dim=1)
            acc = (ind == test_Y).float().mean()

            # Print current data
            print('acc', acc)
            print('loss',test_loss.item())

            # Append acc/loss to plot
            accuracies.append(acc)
            losses.append(test_loss.item())
            steps.append(i)

            if args.importance_sampler:
                show(test_X, test_mask, data)


            with open(f'csv/csv_{modelname_id}.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow([i, acc, test_loss])
            model.train()

    torch.save(model.state_dict(), f'models/model_{modelname_id}.p')
    fig = plt.figure()
    plt.plot(steps, accuracies)
    plt.savefig(f'plots/acc_{modelname_id}.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(steps, losses)
    plt.savefig(f'plots/loss_{modelname_id}.png')
    plt.close(fig)


def show(x, mask, target, data):

    if mask is not None:
        keys = [key for key in data.char2int]

        for l in range(2):
            mask_par = ''
            orig_par = ''
            par = ''
            for i,j in zip(x[l,:],mask[l,:]):
                if keys[j] == keys[i]:
                    par = par + colored(keys[i], 'green')
                else:
                    par = par + colored(keys[i], 'red')

            print(f'batch {l}')
            print(par)

    # else:
    #     print(x)


def accuracy(predictions, targets):
    prediction = predictions.argmax(dim=1)
    target = targets

    accuracy = (target == prediction).float().mean()

    return accuracy.item()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=20000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    parser.add_argument('--load_PATH', type=str, default=None, help="Load model from")

    parser.add_argument('--seed', type=int, default=42, help="Set seed")
    parser.add_argument('--evaluate_steps', type=int, default=25, help="Evaluate model every so many steps")

    parser.add_argument('--eval', type=bool, default=False, help="Evaluate model")
    parser.add_argument('--embedding_size', type=int, default=256, help='Size of the character embeddings used.')

    parser.add_argument('--importance_sampler', type=bool, default=False, help='Boolean representing whether or not to use the importance sampler.')
    args = parser.parse_args()

    # Train the model
    if args.eval:
        evaluate.evaluate(args)
    else:
        train(args)
