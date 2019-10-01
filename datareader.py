import random
from collections import Counter

class DataSet:
    def __init__(self, xfile="../data/wili-2018/x_train.txt",
                       yfile="../data/wili-2018/y_train.txt",
                       x_test="../data/wili-2018/x_test.txt",
                       y_test="../data/wili-2018/y_test.txt"):
        self.par2lan = {}
        self.test_par2lan = {}
        self.languages = set()
        self.test_paragraphs = []
        self.paragraphs = []
        self.lan2pars = {}
        self.epoch_index = 0
        with open(xfile, 'r') as xf, open('latinlangs.txt', 'r') as latinlangs, open(x_test, 'r') as xt:

            # Read in latin languages and strip
            readlatinlangs = latinlangs.readlines()
            readlatinlangs = [lan.strip() for lan in readlatinlangs]

            with open(yfile, 'r') as yf, open(y_test, 'r') as yt:
                for paragraph, label in zip(xf, yf):
                    label = label.rstrip()

                    if label in readlatinlangs:
                        self.languages.add(label)
                        self.par2lan[paragraph] = label
                        self.paragraphs.append(paragraph)

                        try:
                            self.lan2pars[label].append(paragraph)
                        except:
                            self.lan2pars[label] = [paragraph]

                for paragraph, label in zip(xt, yt):
                    label = label.rstrip()

                    if label in readlatinlangs:
                        self.test_par2lan[paragraph] = label
                        self.test_paragraphs.append(paragraph)

                        try:
                            self.lan2pars[label].append(paragraph)
                        except:
                            self.lan2pars[label] = [paragraph]

        self.languages = sorted(list(self.languages))
        self.char2int, self.par2lan, self.all_real_chars, self.all_bigram_chars = self.parse_chars()
        self.lan2int = self.parse_lan()
        print("data reading complete")


    def parse_chars(self, cutoff=1000):
        # chars = [char for par in data.paragraphs for char in par]
        # chars_set = set(chars)

        unk_char = '☃'
        all_real_chars = set()
        all_real_chars.add(unk_char)

        counts = Counter(''.join(self.paragraphs))
        for char in counts:
            if counts[char] >= cutoff:
                all_real_chars.add(char)

        newpar2lan = {}
        for i, par in enumerate(self.paragraphs):
            newpar = []
            for char in par:
                if char in all_real_chars:
                    newpar.append(char)
                else:
                    newpar.append(unk_char)
            self.paragraphs[i] = ''.join(newpar)
            newpar2lan[self.paragraphs[i]] = self.par2lan[par]

        charsandcounts = [(counts[char], char) for char in all_real_chars]
        charsandcounts.sort(reverse=True)
        char2int = {}
        for i, (_, char) in enumerate(charsandcounts):
            char2int[char] = i

        all_bigram_chars = []
        for char1 in sorted(all_real_chars):
            for char2 in sorted(all_real_chars):
                all_bigram_chars.append(char1 + char2)

        return char2int, newpar2lan, all_real_chars, all_bigram_chars

    def parse_lan(self):
        lan2int = {}

        for i, lan in enumerate(self.languages):
            lan2int[lan] = i
        return lan2int

    def get_next_batch(self, batch_size, legal_langs=None):
        batch = []
        targets = []
        while len(batch) < batch_size:
            par = self.paragraphs[self.epoch_index]
            if not legal_langs or self.par2lan[par] in legal_langs:
                batch.append(par)
                targets.append(self.par2lan[par])
            self.epoch_index = (self.epoch_index+1)%len(self.paragraphs)
            if not self.epoch_index:
                random.shuffle(self.paragraphs)
        return batch, targets

    def get_next_test_batch(self, batch_size, legal_langs=None):
        batch = []
        targets = []
        while len(batch) < batch_size:
            par = self.test_paragraphs[self.epoch_index]
            if not legal_langs or self.test_par2lan[par] in legal_langs:
                batch.append(par)
                targets.append(self.test_par2lan[par])
            self.epoch_index = (self.epoch_index+1)%len(self.test_paragraphs)
            if not self.epoch_index:
                random.shuffle(self.paragraphs)
        return batch, targets

if __name__ == '__main__':
    data = DataSet()
    print(data.char2int)
