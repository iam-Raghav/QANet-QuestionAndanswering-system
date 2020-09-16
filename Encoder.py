from __future__ import print_function
import torch as t
import torch.nn as nn
from utils import *

class Encoder_block(nn.Module):
    def __init__(self,args,d_model):
        super(Encoder_block,self).__init__()
        self.pos_enc = PositionalEncoder(d_model)
        self.conv_model = Convol_block(args,d_model)
        if args.layernorm == 'cus':
            self.norm1 = Norm(d_model)
            self.norm2 = Norm(d_model)
        elif args.layernorm == 'inblt':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)


        self.multihead_attn = MultiHeadAttention(args.attention_heads,d_model)

        self.feedfwd = FeedForward(d_model)

    def forward(self,x_embedded,mask = None,blow_up ='N'):
        # input x_embedded coming as (batchsize,maxseqlen,d_model)
        #1. Positional encoding(refer block diagram)
        x_residue = self.pos_enc(x_embedded,blow_up)

        #2.Convolution Block (refer block diagram)
        # Param size :- batchsize,maxseqlen,d_model==> batchsize,maxseqlen,d_model
        x_embedded = self.conv_model(x_residue)
        #residual connection
        # Param size :- batchsize,maxseqlen,d_model ==> batchsize,maxseqlen,d_model
        x_embedded = x_residue + x_embedded

        #3. Self multi attention block (refer block diagram)
        x_residue = x_embedded
        #Normalize the input
        # Param size :- batchsize,maxseqlen,d_model ==> batchsize,maxseqlen,d_model
        x_embedded = self.norm1(x_embedded)
        #calling multi attention
        #Param size :- batchsize, maxseqlen, d_model == > batchsize, maxseqlen, d_model
        x_attended = self.multihead_attn(x_embedded,x_embedded,x_embedded,mask)
        # residual connection
        # Param size :- batchsize,maxseqlen,d_model ==> batchsize,maxseqlen,d_model
        x_attended = x_attended + x_residue

        #4.Feed Forward block (refer block diagram)
        x_residue =x_attended
        #Normalize the input
        # Param size :- batchsize,maxseqlen,d_model ==> batchsize,maxseqlen,d_model
        x_attended = self.norm2(x_attended)
        #calling Feed forward
        #Param size :- batchsize, maxseqlen, d_model == > batchsize, maxseqlen, d_model
        x_attended = self.feedfwd(x_attended)
        # residual connection
        # Param size :- batchsize,maxseqlen,d_model ==> batchsize,maxseqlen,d_model
        x_attended = x_attended + x_residue

        return x_attended

















