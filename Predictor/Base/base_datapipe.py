from tqdm import tqdm
import gensim
import ipdb
import pickle as pk


class BaseDataPipe(object):
    """
    --predict_pipe(self, list_of_lines_str)
    --line_to_middle(self, line_str)
    --line_to_processed(self, line_token)
    --train_w2v(self, sentance_list, embedding_dim, min_count, num_works)
    """
    def __init__(self, tokenizer, vocab=None):
        self.tokenizer = tokenizer
        self.bos, self.eos = '<BOS>', '<EOS>'
        self.vocab = vocab
        self.sentance_list = []

    def predict_pipe(self, list_of_lines_str):
        assert self.vocab is not None
        list_of_lines_id = [self._convert_line(self._clean_split_line(line_str)) for line_str in list_of_lines_str]
        return list_of_lines_id

    def check_process(self):
        raise NotImplementedError

    def line_to_middle_train(self, line_str, is_train=True):
        line_token = self.line_to_middle(line_str, is_train)
        return line_token

    def line_to_middle_dev(self, line_str, is_train=False):
        line_token = self.line_to_middle(line_str, is_train)
        return line_token

    def line_to_middle(self, line_str, is_train=False):
        line_token = self._clean_split_line(line_str, is_train)
        return line_token

    def train_w2v(self, embedding_dim, min_count, num_works):
        """
        train w2v model with list of sentance
        :param sentance_list:
        :return:
        """
        sentance = Sentance(self.sentance_list, self.bos, self.eos)
        model = self._train_w2v(embedding_dim=embedding_dim, min_count=min_count, num_works=num_works, sentance=sentance)
        return model, sentance

    def line_to_processed(self, line_token):
        line_id = self._convert_line(line_token)
        return line_id

    def _clean_split_line(self, line_str, is_train):
        line_str = self._clean_line(line_str)
        line_token = self._tokenize(line_str)
        line_token = self._clean_line_token_post(line_token)
        if is_train:
            self.sentance_list.append(line_token)
        return line_token

    def _clean_line(self, word_str):
        """
        word_str = word_str.lower()
        return word_str
        """
        raise NotImplementedError

    def _clean_line_token_post(self, line_token):
        """

        :param line_token:
        :return:
        return line_token
        """
        raise NotImplementedError

    def _tokenize(self, line_str):
        """
        line_token = self.tokenizer.tokenize(line_str)
        :param line_str:
        :return:
        """
        raise NotImplementedError

    def _convert_line(self, line_token):
        line_id = [self.vocab.from_token_id(i) for i in line_token]
        return line_id

    def _train_w2v(self, embedding_dim, min_count, num_works, sentance):
        model = gensim.models.FastText(size=embedding_dim, min_count=min_count, workers=num_works)
        model.build_vocab(sentance)
        print(f'building vocab')
        model.train(sentance, total_examples=model.corpus_count, epochs=model.iter)
        return model

    def get_vocab(self, vocab):
        self.vocab = vocab

    def id2word_line(self, id_line):
        word_line = [self.vocab.from_id_token(id) for id in id_line]
        return word_line


class Sentance():
    """
    corpus: [['a', 'b', 'c'],]
    """

    def __init__(self, corpus, bos, eos):
        self.corpus = corpus
        self.bos = bos
        self.eos = eos

    def __iter__(self):
        for i in tqdm(self.corpus, desc='itering corpus'):
            yield [self.bos] + i + [self.eos]












