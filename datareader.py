import random
from collections import Counter

class DataSet:
    def __init__(self, xfile="../data/wili-2018/x_train.txt",
                       yfile="../data/wili-2018/y_train.txt"):
        self.par2lan = {}
        self.languages = set()
        self.paragraphs = []
        self.lan2pars = {}
        self.epoch_index = 0
        with open(xfile, 'r') as xf, open('latinlangs.txt', 'r') as latinlangs:

            # Read in latin languages and strip
            readlatinlangs = latinlangs.readlines()
            readlatinlangs = [lan.strip() for lan in readlatinlangs]

            with open(yfile, 'r') as yf:
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

        self.char2int, self.par2lan, self.all_real_chars = self.parse_chars()
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
        return char2int, newpar2lan, all_real_chars

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

if __name__ == '__main__':
    data = DataSet()
    print(data.char2int)
