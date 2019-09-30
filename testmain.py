from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
from testmodel import modelRNN
from vanille_rnn import VanillaRNN
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def retrain(args):
    data = DataSet()

    model = modelRNN(len(data.char2int), len(data.languages))
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    accuracies = []
    for i in range(args.training_steps):

        batch, targets = data.get_next_batch(args.batch_size)

        # make one-hot encoding
        X = torch.zeros(size=(args.batch_size, 140, len(data.char2int)))
        for idx, par in enumerate(batch):
            par = par[-140:]
            for idx2, char in enumerate(par):
                X[idx, idx2, data.char2int[char]] = 1
        
        # get targets in non-one-hot format
        y=[]
        for target in targets:
            y.append(data.lan2int[target])
        y_tensor = torch.tensor(y)


        optimizer.zero_grad()
        out, hidden = model.forward(X.float())
        # we only want the final output
        out = out[:,-1,:].squeeze()
        loss_step = loss(out, y_tensor)
        losses.append(loss_step.item())
        accuracies.append(accuracy(out, y_tensor))
        loss_step.backward()
        optimizer.step()

        if i % 10 == 0:
            acc = accuracy(out, y_tensor)
            print(i)
            print("accuracy:", sum(accuracies[-10:]) / min(10, len(accuracies)))
            print('loss:', sum(losses[-10:]) / min(10, len(losses)))
            print()
    plt.plot(range(len(losses)), losses)
    plt.title('loss over time')
    plt.show()
    
    plt.plot(range(len(accuracies)), accuracies)
    plt.title('accuracy over time')
    plt.show()



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
    parser.add_argument('--num_hidden', type=int, default=133, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='Learning rate')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    # Train the model
    retrain(args)
