from Predictor.Base import BaseDataPipe
from nltk import CoreNLPParser
import re


class ByteDataPipe(BaseDataPipe):
    def __init__(self, tokenizer=CoreNLPParser()):
        super(ByteDataPipe, self).__init__(tokenizer)
        self.tokenizer = tokenizer

    def _clean_line(self, word_str):
        word_str = re.sub()
        word_str = re.sub(r'\([^()]*\)', '', word_str)
        word_str = re.sub(r'\（[^（）]*\）', '', word_str)
        word_str = re.sub(r'\[[^]]*\]', '', word_str)
        word_str = re.sub(r'\{[^{}]*\}', '', word_str)
        word_str = re.sub(r'\{[^{}]*\}', '', word_str)
        word_str = re.sub(r'\【[^【】]*\】', '', word_str)
        return word_str

    def _tokenize(self, line_str):
        line_token = list(self.tokenizer.tokenize(line_str))
        return line_token

    def _clean_line_token_post(self, line_token):
        return line_token


