import numpy as np
import json
import nltk



class DataPipe(object):

    def __init__(self, args, vocab=None):
        self.args = args
        self.vocab = vocab

    def predict_pipe(self):
        pass

    def clean_lines(self):
        pass

    def clean_line(self):
        pass

    def convert_lines(self):
        assert self.vocab is not None

    def convert_line(self):
        pass


