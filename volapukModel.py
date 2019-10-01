import torch
import torch.nn as nn


class VolapukModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, padding_index=0, hidden_size=64, num_layers=1, batch_first=True):

        super(VolapukModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=False)

        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, num_output)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)

        pack = nn.utils.rnn.pack_sequence(embed)
        # print('pack',pack[0].shape,pack[1].shape)
        out, _ = self.lstm(pack, None)
        # print('out',out[0].shape,out[1].shape)
        unpack, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # print('unpack',unpack[0].shape, unpack[1].shape)
        out = unpack[torch.arange(0, x.size(0)).long(), seq_lengths - 1, :]
        # print('out',out.shape)

        out = self.batchnorm(out)

        out = self.linear(out)

        return out
