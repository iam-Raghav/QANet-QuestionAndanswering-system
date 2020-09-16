from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import Embedding as embed
import Encoder as encode
from utils import Linear
import copy
import gc
import os
import sys
import psutil

class Transfrmr_bidaf(nn.Module):
    def __init__(self,args,pretrained):
        super(Transfrmr_bidaf,self).__init__()
        self.embed = embed.Embedding(args,pretrained)

        # Encoder module
        self.encoder_ctxt = encode.Encoder_block(args,2 * args.word_dim)
        self.encoder_ques = encode.Encoder_block(args,2 * args.word_dim)

        #Attention Flow Layer
        self.att_weight_c = Linear(args.hidden_size * 2, 1, args.dropout)
        self.att_weight_q = Linear(args.hidden_size * 2, 1, args.dropout)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1, args.dropout)
        self.N =args.Model_encoder_size
        self.dropout = nn.Dropout(p= args.dropout)

        #Model Encoding Layer
        self.Model_encoder = self.get_clones(encode.Encoder_block(args,8 * args.word_dim), args.Model_encoder_size)
        # self.Model2start= Linear(16 * args.word_dim, 8 * args.word_dim,args.dropout)
        # self.Model2end = Linear(16 * args.word_dim, 8 * args.word_dim,args.dropout)
        # self.start_idx = Linear(16 * args.word_dim,1,args.dropout)
        # self.end_idx = Linear(16 * args.word_dim, 1, args.dropout)
        self.start_idx = nn.Linear(16 * args.word_dim,1)
        self.end_idx = nn.Linear(16 * args.word_dim, 1)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])








    def forward(self,batch):
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # cq_tiled = c_tiled * q_tiled
            # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
            

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def Modelencoders(x):
            for i in range(self.N):
                x = self.Model_encoder[i](x)
            return x



        #Embedding Block
        cxt,quest = self.embed(batch)
        #cxt embed ==> (batch, seq_len, hidden_size * 2)
        #quest_embed ==> (batch, seq_len, hidden_size * 2)

        #Encoder Block
        cxt_mask = (batch.c_word[0] != 1)
        quest_mask = (batch.q_word[0] != 1)
        cxt = self.encoder_ctxt(cxt,mask = cxt_mask, blow_up ='Y')
        quest = self.encoder_ques(quest, mask = quest_mask, blow_up ='Y')
        #del cxt_mask
        #del quest_mask
        #gc.collect()

        #attention flow layer
        cq_attended = att_flow_layer(cxt,quest)
        
        

        #Model Encoding Layer
        M1 = Modelencoders(cq_attended)
        # gc.collect()
        # cpuStats()
        # memReport()
        M2 = Modelencoders(M1)
        M3 = Modelencoders(M2)

        #output layer
        M12 = torch.cat([M1,M2], dim= -1)
        M13 = torch.cat([M1,M3], dim= -1)
        #
        

        # M12 = F.relu(self.Model2start(M12))
        # M13 = F.relu(self.Model2end(M13))

        # M13 = M12 + M13

        M12 = self.start_idx(self.dropout(M12)).squeeze(-1)
        M13 = self.end_idx(self.dropout(M13)).squeeze(-1)

        #p1 = F.log_softmax(M12,dim=-1)
        
        #p2 = F.log_softmax(M13, dim=-1)

        

        return M12 , M13

















