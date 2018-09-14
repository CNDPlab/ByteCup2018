import torch as t
import os
from Predictor import Models




class Summarizer(t.nn.Module):
    def __init__(self, load_from_exp, vocab, args):
        self.vocab = vocab
        self.args = args
        self.model = getattr(Models, args.model_name)(matrix=self.vocab.matrix, args=self.args)
        self.load(load_from_exp)

    def load(self, load_from_exp):
        best_model_path = self.get_best_k_model_path(load_from_exp, k=1)[0]
        trainner_state = t.load(os.path.join(load_from_exp, best_model_path, 'trainner_state'))
        self.model.load_state_dict(t.load(os.path.join(load_from_exp, best_model_path, 'model')))
        self.global_step = trainner_state['epoch']
        self.global_epoch = trainner_state['step']
        self.optim = trainner_state['optim']

    def get_best_k_model_path(self, path, k=1):
        k_best_model_folder = sorted(os.listdir(path), key=lambda x: x.split('T')[1], reverse=True)[:k]
        return k_best_model_folder

    def summarize(self, context_str):


