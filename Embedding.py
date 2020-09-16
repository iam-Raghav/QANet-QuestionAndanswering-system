from __future__ import print_function
import gc
import torch
import torch.nn as nn
from utils import *

class Embedding(nn.Module):
    def __init__(self,args, pretrained):
        super(Embedding, self).__init__()
        # 1. Character Embedding Layer
        self.char_emb_mode = args.char_embed_mode

        self.char_emb_mode = args.char_embed_mode
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        if self.char_emb_mode == 'conv':
            # self.char_conv = nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width))
            self.emb_depthwise = nn.Conv1d(in_channels=args.char_dim, out_channels=args.char_dim, kernel_size=args.char_channel_width,
                                       padding=2, groups=args.char_dim)
            self.emb_pointwise = nn.Conv1d(in_channels=args.char_dim, out_channels=args.char_channel_size, kernel_size=1)           

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        self.dropout = nn.Dropout(p=args.dropout)
        self.args = args

        # highway network
        assert args.hidden_size * 2 == (args.char_channel_size + args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

    def forward(self,batch):
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3).contiguous()
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3))
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.emb_depthwise(x)
            x = F.relu(self.emb_pointwise(x))
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        # 1. Character Embedding Layer
        if self.char_emb_mode == 'conv':
            c_char = char_emb_layer(batch.c_char)
            q_char = char_emb_layer(batch.q_char)
        elif self.char_emb_mode == 'max':
            c_char = self.char_emb(batch.c_char)
            idx = c_char.argmax(-2)
            c_char = c_char.gather(-2,idx.unsqueeze(-2)).squeeze(-2)

            q_char = self.char_emb(batch.q_char)
            idx = q_char.argmax(-2)
            q_char = q_char.gather(-2, idx.unsqueeze(-2)).squeeze(-2)
            idx=0

        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        # c_lens = batch.c_word[1]
        # q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        

        return c , q

