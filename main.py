from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter


def train(args):

    data = DataSet()
    parse_chars(data)

    batch = data.get_next_batch(25)

    for _ in range(args.training_steps):

        batch, targets = data.get_next_batch(args.batch_size)

        # Create one hot array of characters?
        break

def accuracy(predictions, targets):
    pred = np.argmax(predictions, axis=1)
    tar = np.argmax(targets, axis=1)

    return (pred - tar).mean()

def one_hot_encode(batch):
    pass

def parse_chars(data, cutoff=1000):
    # chars = [char for par in data.paragraphs for char in par]
    # chars_set = set(chars)

    unk_char = '☃'
    all_real_chars = set()
    all_real_chars.add(unk_char)
    counts = Counter(''.join(data.paragraphs))
    for char in counts:
        if counts[char] >= cutoff:
            all_real_chars.add(char)

    for i, par in enumerate(data.paragraphs):
        newpar = []
        for char in par:
            if char in all_real_chars:
                newpar.append(char)
            else:
                newpar.append(unk_char)
        data.paragraphs[i] = ''.join(newpar) 

    charsandcounts = [(counts[char], char) for char in all_real_chars]
    charsandcounts.sort()
    char2int = {}
    for i, (char, _) in enumerate(charsandcounts):
        char2int[char] = i
    return char2int

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=128, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    # Train the model
    train(args)
