from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
from model import modelRNN
import torch.nn as nn
from scipy.spatial import distance

def baseline(args):
    data = DataSet()

    chars_ratio = get_rations(data)

    languages = sorted(data.languages)
    all_real_chars = sorted(data.all_real_chars)

    correct = 0

    for pars in data.paragraphs:
        unk_counter = 0

        counters = np.zeros((len(all_real_chars)))

        # Count characters from paragraph, add counts to zero array
        for char in set(pars):
            count = pars.count(char)
            try:
                counters[all_real_chars.index(char)] = count
            except ValueError:
                counters[all_real_chars.index('☃')] += count

        # Get character ratio
        ratio = (counters / len(pars)) * 100

        pred_lan = determine_eucl_dist(ratio, chars_ratio)

        if pred_lan == data.par2lan[pars]:
            correct += 1

        print('\n\npredicted')
        print(pred_lan)
        print('actual', data.par2lan[pars])

    accuracy = correct / len(data.paragraphs)

    print("Accuracy:", accuracy)

def determine_eucl_dist(ratio, chars_ratio):
    smallest = 9999999
    pred_lan = ''
    for lan in chars_ratio.keys():
        dist = distance.euclidean(ratio, chars_ratio[lan])
        if dist < smallest:
            smallest = dist
            pred_lan = lan
    return pred_lan

def get_rations(data):
    """
    Calculates the precence ratios of each character in all paragrpahs of a language.

    Args:
        data: dataset

    Returns:
        chars_ratio: dictionary containing the ratio of each chacater.
                    Value of dict is a np array conainting rations. Index is sorted (alphabetized)
                    characers from data.all_real_chars.
    """
    # Sort languages and characters
    chars_ratio = {}
    languages = sorted(data.languages)
    all_real_chars = sorted(data.all_real_chars)

    # Iterate over languages
    for lan in languages:
        chars_in_lan = []
        pars = ' '.join(data.lan2pars[lan])
        unk_counter = 0

        counters = np.zeros((len(all_real_chars)))

        # Count characters from paragraph, add counts to zero array
        for char in set(pars):
            count = pars.count(char)
            try:
                counters[all_real_chars.index(char)] = count
            except ValueError:
                counters[all_real_chars.index('☃')] += count


        # Get character ratio
        chars_ratio[lan] = (counters / len(pars)) * 100

    return chars_ratio


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=146, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers in the model')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--training_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    # Train the model
    baseline(args)
