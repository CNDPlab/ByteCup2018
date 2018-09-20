from Predictor.Base import BaseDataPipe
from nltk import CoreNLPParser
import nltk
import re
import ipdb


class ByteDataPipe(BaseDataPipe):

    def __init__(self, tokenizer=CoreNLPParser()):
        super(ByteDataPipe, self).__init__(tokenizer)
        self.tokenizer = tokenizer
        self.partten_1 = re.compile('[a-z]\.[A-Z]')
        self.partten_2 = re.compile('[a-z][A-Z]')

    def _clean_line(self, word_str):
        word_str = re.sub('’', '\'', word_str)
        word_str = re.sub('\`\`', '"', word_str)
        word_str = re.sub(r'“', '"', word_str)
        word_str = re.sub(r'”', '"', word_str)
        word_str = re.sub(r'\'\'', '"', word_str)
        word_str = re.sub(r'\([^()]*\)', '', word_str)
        word_str = re.sub(r'\（[^（）]*\）', '', word_str)
        word_str = re.sub(r'\[[^]]*\]', '', word_str)
        word_str = re.sub(r'\{[^{}]*\}', '', word_str)
        word_str = re.sub(r'\【[^【】]*\】', '', word_str)
        return word_str

    def _tokenize(self, line_str):
        line_token = list(self.tokenizer.tokenize(line_str))
        return line_token

    def _clean_line_token_post(self, line_token):
        n_line_token = []
        for token in line_token:
            if re.search(self.partten_1, token) is not None:
                sub_token = re.search(self.partten_1, token)
                n_line_token.append(token[:sub_token.start()+1].lower())
                n_line_token.append('.')
                n_line_token.append(token[sub_token.end()-1:].lower())

            elif re.search(self.partten_2, token) is not None:
                sub_token = re.search(self.partten_2, token)
                n_line_token.append(token[:sub_token.start()+1].lower())
                n_line_token.append('.')
                n_line_token.append(token[sub_token.end()-1:].lower())

            elif token == 'n\'t':
                n_line_token.append('not')
            elif token == '\'ll':
                n_line_token.append('will')
            elif token == '\'re':
                n_line_token.append('are')
            else:
                n_line_token.append(token.lower())


        #TODO UnderstandBusiness \ n’t'


        return n_line_token

"""
todo:
action.Failing
UnderstandBusiness
n’t'

"""