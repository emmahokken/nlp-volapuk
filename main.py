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

def retrain(args):
    data = DataSet()

    # HARDCODED WATCH THIS SPACE
    model = modelRNN(1, args.num_hidden, args.num_layers)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    for i in range(args.training_steps):

        batch, targets = data.get_next_batch(args.batch_size)
        list_of_one_hot_X = []

        for par in batch:
            par = par[-140:]
            x = []
            for char in par:
                x.append(data.char2int[char])
            x_tensor = torch.tensor(x).view(-1, 1)
            # x_one_hot = torch.zeros(x_tensor.size()[0], len(data.char2int)).scatter_(1, x_tensor, 1)
            list_of_one_hot_X.append(x_tensor)

        X = torch.stack(list_of_one_hot_X)

        # convert targets to Y
        languages = list(data.languages)
        sorted(languages)
        y = []
        for target in targets:
            y.append(languages.index(target))
        y_tensor = torch.tensor(y).view(-1, 1)
        Y = torch.zeros(y_tensor.size()[0], len(data.languages)).scatter_(1, y_tensor, 1)

        optimizer.zero_grad()

        out, hidden = model.forward(X.float())

        output = loss(out.float(), Y.long())
        output.backward()
        optimizer.step()

        if i % 100 == 0:
            acc = accuracy(out, y_tensor)
            print("accuracy:", acc)
            print('loss:', output.item())
            print()

def train(args):

    timestamp = time.asctime(time.localtime(time.time())).replace('  ',' ').replace(' ','_')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)
    
    data = DataSet()
    # parse_chars(data)

    vocab_size = 217
    args.embedding_size = 128
    args.classes = 146
    args.rnn = 'LSTM'
    args.mean_seq = False
    args.hidden_size = 128
    args.layers = 2
    # model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes,
    #         hidden_size=args.hidden_size, num_layers=args.layers, batch_first=True).to(device=device)
    model = VolapukModel(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes,
            hidden_size=args.hidden_size, num_layers=args.layers, batch_first=True).to(device=device)


    if args.load_model:
        print(f'Load model {args.PATH}.p')
        model.load_state_dict(torch.load(f'{args.PATH}.p'))

    loss = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    temp_batch_size = args.batch_size

    losses = []
    accuracies = []
    steps = []

    # with open('latinlangs.txt', 'r') as latinlangs:
    #     train_languages = latinlangs.readlines()
    #     train_languages = [tl[:-1] for tl in train_languages]

    # print('train_languages', train_languages)
    print(data.languages)

    for i in tqdm(range(args.training_steps)):

        # get batch and targets, however not in correct format
        batch, targets = data.get_next_batch(args.batch_size)
        list_of_one_hot_X = []

        # Convert batch to X
        y = []
        languages = list(data.languages)
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
            # x_one_hot = torch.zeros(x_tensor.size()[0], len(data.char2int)).scatter_(1, x_tensor, 1)
            # list_of_one_hot_X.append(x_one_hot)

            # if target in train_languages:
            #     list_of_one_hot_X.append(x_tensor)
            #     y.append(languages.index(target))
            list_of_one_hot_X.append(x_tensor)
            y.append(languages.index(target))

        X = torch.stack(list_of_one_hot_X,dim=0).squeeze()
        Y = torch.tensor(y).type(torch.LongTensor).view(-1, 1).squeeze()

        # print(Y.shape)
        args.batch_size = Y.shape[0]

        X = X.to(device)
        Y = Y.to(device)
        
        out = model.forward(X, (torch.ones(args.batch_size)*args.batch_size).long()) # lstm from dl
        optimizer.zero_grad()
        output = criterion(out, Y)
        output.backward()
        optimizer.step()

        if i % args.evaluate_steps == 0:
            losses.append(output.item())
            _, ind = torch.max(out, axis=1)
            acc = (ind == Y).float().mean()
            print('acc', acc)
            print('loss',output.item())
            accuracies.append(acc)
            steps.append(i)

    torch.save(model.state_dict(), f'{args.PATH}__it{args.training_steps}_seed{args.seed}.p')
    torch.save(model.state_dict(), f'{args.PATH}_{timestamp}.p')
    # torch.save(data, f'{args.PATH}_data')
    fig = plt.figure()
    plt.plot(steps, accuracies)
    # plt.show()
    plt.savefig(f'plots/acc_model_it{args.training_steps}_seed{args.seed}.png')
    plt.savefig(f'plots/acc_{timestamp}.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(steps, losses)
    # plt.show()
    plt.savefig(f'plots/loss_model_it{args.training_steps}_seed{args.seed}.png')
    plt.savefig(f'plots/loss_{timestamp}.png')
    plt.close(fig)


def accuracy(predictions, targets):
    # pred = torch.argmax(predictions, dim=1).float()
    # tar = targets.float()
    #
    # return (pred - tar).mean()

    prediction = predictions.argmax(dim=1)
    target = targets

    accuracy = (target == prediction).float().mean()

    return accuracy.item()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--PATH', type=str, default="models/model", help="Model name to save")
    parser.add_argument('--load_model', type=bool, default=False, help="Load model from PATH")

    parser.add_argument('--seed', type=int, default=42, help="Set seed")
    parser.add_argument('--evaluate_steps', type=int, default=25, help="Evaluate model every so many steps")

    args = parser.parse_args()

    # Train the model
    # retrain(args)
    train(args)
