from configs import DefaultConfigs
from torch.utils.data import DataLoader
import torch as t
import fire
from Predictor.Utils import Vocab
from DataSets import get_dataloader
import pickle as pk
from Trainner import Trainner
from Predictor import Models
from Predictor.Utils.LossFunctions import loss_function
from Predictor.Utils.ScoreFunctions import score_function
from Predictor import Predictor
import os
import ipdb
from flask import Flask, request
app = Flask(__name__)




def train(**kwargs):
    args = DefaultConfigs()
    args.parse(kwargs)
    train_loader = get_dataloader(os.path.join(args.byte_root, 'train'))
    dev_loader = get_dataloader(os.path.join(args.byte_root, 'dev'))
    vocab = pk.load(open(os.path.join(args.vocab_folder, args.vocab_name), 'rb'))
    args.eos_id, args.sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    model = getattr(Models, args.model_name)(matrix=vocab.matrix, args=args)
    trainner = Trainner(args, vocab, model, loss_function, score_function, train_loader, dev_loader)
    trainner.init_trainner()
    trainner.train()


def test(**kwargs):
    args = DefaultConfigs()
    args.parse(kwargs)
    assert args.load_from_exp is not None
    test_loader = get_dataloader(os.path.join(args.byte_root, 'test'))
    vocab = pk.load(open(os.path.join(args.vocab_folder + 'vocab.pkl'), 'rb'))
    args.eos_id, args.sos_id = vocab.token2id['<EOS>'], vocab.token2id['<BOS>']
    predictor = Predictor(args.load_from_exp, vocab, args)


@app.route('/', methods=['get'])
def predict():
    message = request.args.get('message')
    message = ' '.join(message.split('_'))
    # args.parse(kwargs)
    # assert args.load_from_exp is not None
    return message


if __name__ == '__main__':
    fire.Fire()
