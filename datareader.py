import random

class DataSet:
    def __init__(self, xfile="./data/wili-2018/x_train.txt",
                       yfile="./data/wili-2018/y_train.txt"):
        self.par2lan = {}
        self.languages = set()
        self.paragraphs = []
        self.epoch_index = 0
        with open(xfile, 'r') as xf:
            with open(yfile, 'r') as yf:
                for paragraph, label in zip(xf, yf):
                    label = label.rstrip()
                    self.languages.add(label)
                    self.par2lan[paragraph] = label
                    self.paragraphs.append(paragraph)
        print("data reading complete")

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
    for i in range(100):
        print(i, data.get_next_batch(1, {'swa', 'rus', 'nld'}))
