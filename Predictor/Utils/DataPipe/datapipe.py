from Predictor.Base import BaseDataPipe
from nltk import CoreNLPParser


class DataPipe(BaseDataPipe):
    """
    --predict_pipe(self, list_of_lines_str)
    --line_to_middle(self, line_str)
    --line_to_processed(self, line_token)
    --train_w2v(self, sentance_list, embedding_dim, min_count, num_works)
    """
    def __init__(self, tokenizer=CoreNLPParser()):
        super(DataPipe, self).__init__(tokenizer)
        self.tokenizer = tokenizer

    def _clean_line(self, word_str):
        """
        word_str = word_str.lower()
        return word_str
        """
        word_str = word_str.lower()
        return word_str

    def _tokenize(self, line_str):
        """
        line_token = self.tokenizer.tokenize(line_str)
        :param line_str:
        :return:  line_token
        """
        line_token = self.tokenizer.tokenize(line_str)
        return line_token

    def _clean_split_line(self, line_str):
        line_str = self._clean_line(line_str)
        line_token = self.tokenizer.tokenize(line_str)
        line_token = self._add_bos_eos(line_token)
        return line_token