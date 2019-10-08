import torch
import torch.nn as nn


class VolapukModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, importance_sampler, padding_index=0, hidden_size=64, num_layers=1, batch_first=True):

        super(VolapukModel, self).__init__()

        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.importance_sampler = importance_sampler

        self.embedding = nn.Embedding(vocab_size+1, embed_size)

        self.lstm = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=False)

        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, num_output)

        self.sigmoid = nn.Sigmoid()
        self.lstm_encoder = nn.LSTM( input_size=embed_size, hidden_size=140, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=False)

    def forward(self, x, seq_lengths):

        embed = self.embedding(x)

        if self.importance_sampler:
            pack = nn.utils.rnn.pack_sequence(embed)
            enc_lstm, _ = self.lstm_encoder(pack,None)
            unpack, _ = nn.utils.rnn.pad_packed_sequence(enc_lstm, batch_first=True)
            enc_lstm = unpack[torch.arange(0, x.size(0)).long(), seq_lengths - 1, :]
            sig = self.sigmoid(enc_lstm)

            bernou = torch.bernoulli(sig)
            mask = bernou.long()
            mask_x = mask*x

            embed = self.embedding(x)
        else:
            mask_x = None
            mask = torch.tensor(0)


        pack = nn.utils.rnn.pack_sequence(embed)

        out, h = self.lstm(pack, None)

        unpack, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = unpack[torch.arange(0, x.size(0)).long(), seq_lengths - 1, :]

        out = self.batchnorm(out)

        out = self.linear(out)

        return out, mask_x, torch.sum(mask).item()
