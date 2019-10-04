from datareader import DataSet
import sys
import numpy as np
import torch
import argparse
from collections import Counter
from model import modelRNN
import torch.nn as nn
from scipy.spatial import distance
from nltk import bigrams
from tqdm import tqdm

def bigram_ratio(data):

    chars_ratio = {}

    languages = sorted(data.languages)

    for lan in tqdm(languages):
        joined_paragraphs = ' '.join(data.lan2pars[lan])
        bigram = list(bigrams(joined_paragraphs))

        counters = np.zeros((len(data.all_bigram_chars)))

        bigram_counts = Counter(bigram)
        summer = 0
        summ = 0
        # Count characters from paragraph, add counts to zero array
        for bi in set(bigram):
            bi_list = list(bi)
            if bi_list[0] not in data.all_real_chars:
                bi_list[0] = '☃'
            if bi_list[1] not in data.all_real_chars:
                bi_list[1] = '☃'
            bi_str = ''.join(bi_list)
            counters[data.all_bigram_chars.index(bi_str)] += bigram_counts[bi] / len(bigram)

        chars_ratio[lan] = counters

    return chars_ratio

def final(args):
    data = DataSet()

    chars_ratio = bigram_ratio(data)

    languages = sorted(data.languages)

    correct = 0

    for pars in tqdm(data.test_paragraphs):
        bigram = list(bigrams(pars))

        counters = np.zeros((len(data.all_bigram_chars)))
        bigram_counts = Counter(bigram)
        # Count characters from paragraph, add counts to zero array
        for bi in set(bigram):
            bi_list = list(bi)
            if bi_list[0] not in data.all_real_chars:
                bi_list[0] = '☃'
            if bi_list[1] not in data.all_real_chars:
                bi_list[1] = '☃'
            bi_str = ''.join(bi_list)
            counters[data.all_bigram_chars.index(bi_str)] += bigram_counts[bi] / len(bigram)

        # print('\nbigram length')
        # print(len(bigram))
        # Get character ratio
        ratio = (counters / len(bigram)) * 100
        # print('ratio')
        # print(ratio)
        pred_lan = determine_eucl_dist(ratio, chars_ratio)

        if pred_lan == data.test_par2lan[pars]:
            correct += 1

        # print('\npred', pred_lan)
        # print('actual', data.test_par2lan[pars])

    accuracy = correct / len(data.test_paragraphs)

    print("Accuracy:", accuracy)

def unigram_baseline(args):
    data = DataSet()

    chars_ratio = get_rations(data)

    languages = sorted(data.languages)
    all_real_chars = sorted(data.all_real_chars)

    correct = 0

    for pars in data.test_paragraphs:
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

        if pred_lan == data.test_par2lan[pars]:
            correct += 1

    accuracy = correct / len(data.test_paragraphs)

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


    parser.add_argument('--unigram', type=bool, default=False)
    args = parser.parse_args()

    # Train the model
    if args.unigram:
        unigram_baseline(args)
    else:
        final(args)
