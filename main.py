import argparse
import copy, json, os

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from data import SQuAD
from model import Transfrmr_bidaf
# from ema import EMA
import evaluate


def train(args, data):
    model_path  = '/content/drive/My Drive/Colab Notebooks/Quesandanswer/saved_models/model.pt'
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = Transfrmr_bidaf(args, data.WORD.vocab.vectors).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    

    # ema = EMA(args.exp_decay_rate)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters,betas=([0.8,0.999]), eps=1e-7, lr=args.learning_rate)
    # optimizer = optim.Adadelta(parameters,  lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    log_dir_path = '/content/drive/My Drive/Colab Notebooks/Quesandanswer/runs/'

    writer = SummaryWriter(log_dir=log_dir_path )  #+ args.model_time

    # model.train()
    loss, last_epoch = 0, -1
    # max_dev_exact, max_dev_f1 = -1, -1

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
            if not present_epoch != 0:
                torch.save(model.state_dict(), model_path)
        last_epoch = present_epoch

        p1, p2 = model(batch)

        optimizer.zero_grad()
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         ema.update(name, param.data)

        if (i + 1) % args.print_freq == 0:

            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            print(f'For every : {args.print_freq: .1f} batch loss per batch : {loss/args.print_freq: .3f} time :{strftime("%H_%M_%S", gmtime())}')


            loss = 0
            # model.train()
    dev_loss, dev_exact, dev_f1 = test(model,  args, data) #test(model, ema, args, data)
    # writer.add_scalar('loss/dev', dev_loss, c)
    # writer.add_scalar('exact_match/dev', dev_exact, c)
    # writer.add_scalar('f1/dev', dev_f1, c)
    print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
          f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

    writer.close()
    # print(f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')

    return model

def test(model, args, data):  #test(model, ema, args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    loss = 0
    answers = dict()
    model.eval()

    # backup_params = EMA(0)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         backup_params.register(name, param.data)
    #         param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = batch.id[i]
                # print(s_idx[i])
                # print(e_idx[i])
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i]+1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         param.data.copy_(backup_params.get(name))

    # writer.add_scalar('loss/dev', dev_loss, c)
    # writer.add_scalar('exact_match/dev', dev_exact, c)
    # writer.add_scalar('f1/dev', dev_f1, c)
    # print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
    #       f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')
    #
    # if dev_f1 > max_dev_f1:
    #     max_dev_f1 = dev_f1
    #     max_dev_exact = dev_exact
    #     best_model = copy.deepcopy(model)

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']


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
    parser.add_argument('--train-batch-size', default=5, type=int)
    parser.add_argument('--train-file', default='train-v2.0.json')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--attention-heads', default=4, type=int)
    parser.add_argument('--Model-encoder-size', default=1, type=int)
    parser.add_argument('--char-embed-mode',default='max')
    parser.add_argument('--layernorm',default='inblt')
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args,'padding_idx',1)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', f'/content/drive/My Drive/Colab Notebooks/Quesandanswer/dataset/{args.dev_file}')
    setattr(args, 'prediction_file', f'/content/drive/My Drive/Colab Notebooks/Quesandanswer/prediction{args.gpu}.out')
    setattr(args, 'model_time', strftime('%H_%M_%S', gmtime()))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('/content/drive/My Drive/Colab Notebooks/Quesandanswer/saved_models'):
        os.makedirs('/content/drive/My Drive/Colab Notebooks/Quesandanswer/saved_models')
    save_path = '/content/drive/My Drive/Colab Notebooks/Quesandanswer/saved_models/model.pt'
    torch.save(best_model.state_dict(), save_path)
    print('training finished!')

if __name__ == '__main__':
    main()
