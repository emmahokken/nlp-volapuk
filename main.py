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
from vae import VAE
from termcolor import colored
import csv

# def retrain(args):
#     data = DataSet()

#     # HARDCODED WATCH THIS SPACE
#     model = modelRNN(1, args.num_hidden, args.num_layers)
#     loss = nn.CrossEntropyLoss()
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

#     for i in range(args.training_steps):

#         batch, targets = data.get_next_batch(args.batch_size)
#         list_of_one_hot_X = []

#         for par in batch:
#             par = par[-140:]
#             x = []
#             for char in par:
#                 x.append(data.char2int[char])
#             x_tensor = torch.tensor(x).view(-1, 1)
#             # x_one_hot = torch.zeros(x_tensor.size()[0], len(data.char2int)).scatter_(1, x_tensor, 1)
#             list_of_one_hot_X.append(x_tensor)

#         X = torch.stack(list_of_one_hot_X)

#         # convert targets to Y
#         languages = list(data.languages)
#         sorted(languages)
#         y = []
#         for target in targets:
#             y.append(languages.index(target))
#         y_tensor = torch.tensor(y).view(-1, 1)
#         Y = torch.zeros(y_tensor.size()[0], len(data.languages)).scatter_(1, y_tensor, 1)

#         optimizer.zero_grad()

#         out, hidden = model.forward(X.float())

#         output = loss(out.float(), Y.long())
#         output.backward()
#         optimizer.step()

#         if i % 100 == 0:
#             acc = accuracy(out, y_tensor)
#             print("accuracy:", acc)
#             print('loss:', output.item())
#             print()

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
        # for all characters in the paragraph, try to append it, else it should be _unk_
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

    params_id = '_b'+str(args.batch_size)+'_h'+str(args.num_hidden)+'_l'+str(args.num_layers)+'_s'+str(args.seed)+'_it'+str(args.training_steps)+'_k'+str(args.k)

    timestamp = time.asctime(time.localtime(time.time())).replace('  ',' ').replace(' ','_')

    modelname_id = params_id+'_'+timestamp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using available device {device}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)
    
    data = DataSet()
    # parse_chars(data)

    # print(len(data.char2int))

    args.vocab_size = len(data.char2int)
    args.embedding_size = 256#len(data.char2int)#128
    args.classes = len(data.languages)
    args.num_hidden = 64
    # args.layers = 2

    print(args)

    model = VolapukModel(vocab_size=args.vocab_size, embed_size=args.embedding_size, num_output=args.classes,
            hidden_size=args.num_hidden, num_layers=args.num_layers, batch_first=True, k=args.k).to(device=device)

    vae = VAE(hidden_dim=500, z_dim=20, input_dim=140, device=device).to(device=device)

    if args.load_PATH is not None:
        print(f'Load model {args.load_PATH}')
        model.load_state_dict(torch.load(f'{args.load_PATH}'))


    print('\n#################################')
    print('############# Model #############')
    print('#################################')
    print(model)

    # loss = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    vae_optimizer = torch.optim.Adam(model.parameters())

    temp_batch_size = args.batch_size

    losses = []
    accuracies = []
    steps = []

    # with open('latinlangs.txt', 'r') as latinlangs:
    #     train_languages = latinlangs.readlines()
    #     train_languages = [tl[:-1] for tl in train_languages]

    print('\n###################################')
    print('############ Languages ############')
    print('###################################')
    print(data.languages)

    for i in tqdm(range(args.training_steps)):

        # get batch and targets, however not in correct format
        batch, targets = data.get_next_batch(args.batch_size)
        X,Y = get_X_Y_from_batch(batch, targets, data, args.device)
        
        # elbo, z, output = vae.forward(X.float())
        # # average_epoch_elbo += loss.item()
        # vae_optimizer.zero_grad()
        # elbo.backward()
        # vae_optimizer.step()
        # print('VAE')

        # print('X',X[0,:])
        # print('z',z[0,:])
        # print('output', output[0,:])
        # print('elbo',elbo)
        lambd = 0.1
        out, _, mask_loss = model.forward(X, (torch.ones(args.batch_size)*args.batch_size).long()) # lstm from dl
        # print(mask_loss, args.batch_size, mask_loss/args.batch_size)
        optimizer.zero_grad()
        loss = criterion(out, Y)
        loss = loss + lambd * mask_loss/args.batch_size
        loss.backward()
        optimizer.step()

        if i % args.evaluate_steps == 0:
            # get batch from test data
            model.eval()
            test_batch, test_targets = data.get_next_test_batch(args.batch_size)
            test_X, test_Y = get_X_Y_from_batch(test_batch, test_targets, data, args.device)
            test_out, test_mask, test_mask_loss = model.forward(test_X, (torch.ones(args.batch_size)*args.batch_size).long())
            test_loss = criterion(test_out, test_Y)
            test_loss = test_loss + lambd * test_mask_loss/args.batch_size

            # get max indices for accuracy
            _, ind = torch.max(test_out, axis=1)
            acc = (ind == test_Y).float().mean()

            # print current data
            print('acc', acc)
            print('loss',test_loss.item())

            # append acc/loss to plot
            accuracies.append(acc)
            losses.append(test_loss.item())
            steps.append(i)

            show(test_X, test_mask, data)

            with open(f'csv/csv_{modelname_id}.csv','a') as f:
                # fd.write()
                writer = csv.writer(f)
                writer.writerow([i, acc, test_loss]
)
            model.train()

    torch.save(model.state_dict(), f'models/model_{modelname_id}.p')
    fig = plt.figure()
    plt.plot(steps, accuracies)
    # plt.show()
    plt.savefig(f'plots/acc_{modelname_id}.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(steps, losses)
    # plt.show()
    plt.savefig(f'plots/loss_{modelname_id}.png')
    plt.close(fig)


def show(x, mask, data):
    # print(x.shape)
    # print(x)
    if mask is not None:
        # print(mask.shape)
        # print(mask)
        # print(mask[0,:])
        # print(data.char2int)
        keys = [key for key in data.char2int]
        # print(x.shape)
        for l in range(2):
        # for l in range(x.shape[0]):
            mask_par = ''
            orig_par = ''
            par = ''
            for i,j in zip(x[l,:],mask[l,:]):
                # mask_par = mask_par + colored(keys[j], 'green')
                # orig_par = orig_par + colored(keys[i], 'red')
                if keys[j] == keys[i]:
                    par = par + colored(keys[i], 'green')
                else:
                    par = par + colored(keys[i], 'red')

            print(f'batch {l}')
            # print(mask_par)
            # print(orig_par)
            print(par)

    else:
        print(x)


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
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # parser.add_argument('--PATH', type=str, default="", help="Model name to save")
    parser.add_argument('--load_PATH', type=str, default=None, help="Load model from")

    parser.add_argument('--seed', type=int, default=42, help="Set seed")
    parser.add_argument('--evaluate_steps', type=int, default=25, help="Evaluate model every so many steps")

    parser.add_argument('--eval', type=bool, default=False, help="Evaluate model")

    parser.add_argument('--k', type=int, default=140, help="Num of characters for prediction")

    args = parser.parse_args()

    # Train the model
    # retrain(args)
    if args.eval:
        data = DataSet()
        
    else:
        train(args)
