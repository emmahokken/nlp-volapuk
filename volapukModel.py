import torch
import torch.nn as nn


class VolapukModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, padding_index=0, hidden_size=64, num_layers=1, batch_first=True, k=140):

        super(VolapukModel, self).__init__()

        self.vocab_size = vocab_size
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(vocab_size+1, embed_size)
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=False)

        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, num_output)

        self.sigmoid = nn.Sigmoid()
        self.lstm_encoder = nn.LSTM( input_size=embed_size, hidden_size=140, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=False)

        # self.mask = torch.nn.Parameter(torch.randn(16,140) / 1000)
        # self.bilinear = nn.Linear(140,140)

    def forward(self, x, seq_lengths):

        importance_sampler = True
        # x = self.bilinear(x)

        # embed = torch.nn.functional.one_hot(x, self.vocab_size).float()
        # print(x.shape)
        embed = self.embedding(x)
        # print(embed)

        if(importance_sampler):
            pack = nn.utils.rnn.pack_sequence(embed)
            enc_lstm, _ = self.lstm_encoder(pack,None)
            unpack, _ = nn.utils.rnn.pad_packed_sequence(enc_lstm, batch_first=True)
            enc_lstm = unpack[torch.arange(0, x.size(0)).long(), seq_lengths - 1, :]
            sig = self.sigmoid(enc_lstm)
            ### prev try
            # bernou = torch.bernoulli(sig)
            # mask = bernou.long()*x
            # # it always gives 0 to the 0 version, should be made better but for now it works??
            # mask[mask == 0] = self.vocab_size+1
            # embed = self.embedding(mask)
            # print(embed)
            
            ### new try
            # k = 100# num of characters to highligh
            v,i = torch.topk(sig,self.k)
            # print(i.is_cuda, x.is_cuda)
            # mask_x = torch.zeros(x.shape).to(self.device).scatter_(1,i,1)
            mask_x = torch.zeros(x.shape).to(self.device).scatter_(1,i.to(self.device),1).long() * x
            embed = self.embedding(x)
        else:
            mask_x = None


        # stop

        # print('shapes',x.shape,embed.shape)
        # print('pre-x',x.shape)
        # print('post-x',embed)
        pack = nn.utils.rnn.pack_sequence(embed)
        # print('pack',pack[0].shape,pack[1].shape)
        out, h = self.lstm(pack, None)
        # print('out',out[0].shape,out[1].shape)
        # print(self.sigmoid(out[0]))
        # print('h',len(h),h[0].shape)
        # for o in range(out[0].shape[0]):
        #     print(out[0][o,:])
        # print(dead)
        unpack, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # print('up',up.shape)
        # print('unpack',unpack[0].shape, unpack[1].shape)
        out = unpack[torch.arange(0, x.size(0)).long(), seq_lengths - 1, :]
        # print('out',out.shape)
        # print('',out.shape)

        out = self.batchnorm(out)
        # print('outbn',out.shape)
        out = self.linear(out)
        # print('outl1',out.shape)

        return out, mask_x

