from datareader import DataSet
import sys
import numpy as np
import torch
import argparse



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

def parse_chars(data):
    # chars = [char for par in data.paragraphs for char in par]
    # chars_set = set(chars)

    final_chars = ['☃']
    counts = dict()
    # chars = []
    for par in data.paragraphs:
        for char in par:
            try:
                counts[char] += 1
            except:
                counts[char] = 0
            if counts[char] > 10000 and char not in final_chars:
                final_chars.append(char)

    # counts = [chars.count(char) for char in chars_set]
    # for char in chars_set:
    #     counting = chars.count(char)
    #     if counting > 100:
    #         final_chars.append(char)

    print(final_chars)
    print(len(final_chars))

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
