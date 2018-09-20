import torch as t
import os
from Predictor import Models
from Predictor.Utils.DataPipe import ByteDataPipe
from nltk.parse.corenlp import CoreNLPParser


class Predictor(t.nn.Module):
    def __init__(self, load_from_exp, vocab, args):
        self.vocab = vocab
        self.args = args
        self.model = getattr(Models, args.model_name)(matrix=self.vocab.matrix, args=self.args)
        self.load(load_from_exp)
        self.data_pipe = ByteDataPipe()
        self.data_pipe.get_vocab(vocab)

    def load(self, load_from_exp):
        best_model_path = self.get_best_k_model_path(load_from_exp, k=1)[0]
        self.model.load_state_dict(t.load(os.path.join(load_from_exp, best_model_path, 'model')))

    def get_best_k_model_path(self, path, k=1):
        k_best_model_folder = sorted(os.listdir(path), key=lambda x: x.split('T')[1], reverse=True)[:k]
        return k_best_model_folder

    def predict(self, list_of_str):
        input_vector = self.data_pipe.predict_pipe(list_of_str)
        with t.no_grad():
            model_outputs = self.model(input_vector)
            output_ids = model_outputs.data.tolist()

        output_token = [self.data_pipe.id2word_line(output_id) for output_id in output_ids]
        return output_token

