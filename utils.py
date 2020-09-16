import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_ff = d_model + 256

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, self.d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, blow_up = 'N'):
        
        if blow_up == 'Y':
        # make embeddings relatively larger
            x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].clone().detach().requires_grad_(False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return x

class Convol_block(nn.Module):
    def __init__(self,args, dmodel):
        super(Convol_block,self).__init__()
        self.dim = dmodel

        if args.layernorm == 'cus':
            self.norm =  Norm(d_model)
        elif args.layernorm == 'inblt':
            self.norm = nn.LayerNorm(dmodel)
        # self.conv_model = nn.Conv1d(1 , self.dim,  args.char_channel_width)
        self.depthwise = nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=args.char_channel_width, padding=2, groups=self.dim)
        self.pointwise = nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=1)

    def forward(self, x):
        #normalize the input
        x = self.norm(x).permute(0,2,1)
        # batch_size = x.size(0)
        # # batch_size,seq_len,2*word-dim ==> batch_size*seq_len,1,2*word-dim
        # x = x.view(-1, 1, x.size(2))
        # # batch_size*seq_len,1,2*word-dim ==> batch_size*seq_len,2*word-dim,conv_length
        # x = self.conv_model(x)
        # # batch_size*seq_len,2*word-dim,conv_length ==> batch_size*seq_len,2*word-dim,1 ==> batch_size*seq_len,2*word-dim
        # x = F.max_pool1d(x, x.size(2)).squeeze()
        # #batch_size * seq_len, 2 * word - dim ==> batch_size,seq_len,2*word-dim
        # x = x.view(batch_size, -1, x.size(1))
        x = self.depthwise(x)
        x = self.pointwise(x).permute(0, 2, 1)
        x = F.relu(x)


        return x

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


