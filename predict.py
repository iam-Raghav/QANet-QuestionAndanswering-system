import argparse
import copy, json, os
import textwrap
import torch
import nltk
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from data import SQuAD
from model import Transfrmr_bidaf

import evaluate

class dummy:
    pass

def word_to_ix(x,data):
    word_idx = [data.WORD.vocab.stoi[word] for word in nltk.word_tokenize(x.lower())]
    char_idx=[]
    max_len = len(max(nltk.word_tokenize(x.lower()), key=len))
    for word in nltk.word_tokenize(x.lower()):
        temp = [data.CHAR.vocab.stoi[letter] for letter in word]
        for i in range(max_len - len(word)):
            temp.append(1)
        char_idx.append(temp)
    c_word = torch.tensor(word_idx, dtype=torch.long)
    c_char = torch.tensor(char_idx, dtype=torch.long)
    return [c_word.unsqueeze(0), ],c_char.unsqueeze(0)

def predict(args, data):
    context =''                     
    question =''
    answer =''
    model = Transfrmr_bidaf(args, data.WORD.vocab.vectors)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    batch = dummy()

    while (1):
        context = input('Input the context paragraph not more than 150 words >')
        if len(context.split()) > 150:
            print('Context is more than 150 words, please input the context less than 150')
        else:
            break
    print('Context:')
    print(textwrap.fill(context, width=100))
    batch.c_word, batch.c_char = word_to_ix(context,data)
    while (1):
        question = input('Ask a Question >')
        batch.q_word, batch.q_char = word_to_ix(question,data)
        temp_context = batch.c_word[0]
        temp_context=temp_context.squeeze(0)
        with torch.set_grad_enabled(False):
            p1, p2 = model(batch)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1,-1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(-1)
            answer = temp_context[s_idx:e_idx + 1]
            answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
            print('Answer>>>')
            print(textwrap.fill(answer, width=60))
            out = input('Want to ask another question Yes(Y) or No(N) >')
            if out == 'N':
                break






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=50, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=150, type=int)
    parser.add_argument('--dev-batch-size', default=5, type=int)
    parser.add_argument('--dev-file', default='dev-v2.0.json')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--train-batch-size', default=50, type=int)
    parser.add_argument('--train-file', default='train-v2.0.json')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--attention-heads', default=4, type=int)
    parser.add_argument('--Model-encoder-size', default=4, type=int)
    parser.add_argument('--char-embed-mode',default='max')
    parser.add_argument('--layernorm',default='inblt')
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args,'padding_idx',1)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'model_path', '/content/drive/My Drive/Colab Notebooks/Quesandanswer/saved_models/model.pt')
    print('data loading complete!')
    predict(args, data)
    print('Good Bye!!!')


if __name__ == '__main__':
    main()

