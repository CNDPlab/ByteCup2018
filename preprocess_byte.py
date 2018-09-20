import os
import json
import pandas as pd
from tqdm import tqdm
from Predictor.Utils.Vocab import Vocab
from sklearn.model_selection import train_test_split
import shutil
from Predictor.Utils.DataPipe import ByteDataPipe
from configs import DefaultConfigs
import fire
import pickle as pk
import ipdb


config = DefaultConfigs()


class DataCollector(object):
    def __init__(self, sample):
        self.data_root = 'Datas/byte_cup/raw/'
        self.prefix = 'bytecup.corpus.train'
        self.files = [file for file in os.listdir(self.data_root) if self.prefix in file]
        self.get_keys()
        self.sample = sample
        if self.sample:
            self.collected_file = 'sample.json'
        else:
            self.collected_file = 'all_data.json'
        self.all_datas = []
        self.dataframes = {'train': None, 'dev': None, 'test': None}

    def get_keys(self):
        with open(os.path.join(self.data_root, self.files[0])) as reader:
            for line in reader:
                self.keys = json.loads(line).keys()
                break

    def collect_files(self):
        if not os.path.exists(os.path.join(self.data_root, self.collected_file)):
            for file in tqdm(self.files, desc='files'):
                self.collect_lines(file)
            self.split_train_test()
            json.dump(self.dataframes, open(os.path.join(self.data_root, self.collected_file), 'w'), ensure_ascii=False)
        else:
            self.dataframes = json.load(open(os.path.join(self.data_root, self.collected_file)))

    def collect_lines(self, file):
        if self.sample:
            with open(os.path.join(self.data_root, file)) as reader:
                for index, line in tqdm(enumerate(reader), desc='lines'):
                    if index > 30:
                        break
                    self.all_datas.append(json.loads(line))
        else:
            with open(os.path.join(self.data_root, file)) as reader:
                for index, line in tqdm(enumerate(reader), desc='lines'):
                    self.all_datas.append(json.loads(line))

    def split_train_test(self):
        self.dataframes['train'], self.dataframes['dev'] = train_test_split(self.all_datas, test_size=0.01, random_state=1)
        self.dataframes['dev'], self.dataframes['test'] = train_test_split(self.dataframes['dev'], test_size=0.5, random_state=1)
        del self.all_datas


class DataProcessor(object):
    def __init__(self, data_collector, min_count):
        assert config.vocab_name is not None
        self.data_collector = data_collector
        self.handler = ByteDataPipe()
        self.vocab = Vocab()
        self.min_count = min_count

    def _process2middle(self):
        for set in self.data_collector.dataframes.keys():
            self._process2middle_set(set)

    def _process2middle_set(self, set='train'):
        assert set in ['train', 'test', 'dev']
        if set == 'train':
            func = self.handler.line_to_middle_train
        else:
            func = self.handler.line_to_middle_dev

        self.data_collector.dataframes[set+'_middle'] = []
        for line in tqdm(self.data_collector.dataframes[set], desc=set+'_middle'):
            middle_line = {}
            middle_line['content'], middle_line['title'], middle_line['id'] = func(line['content']), func(line['title']), line['id']
            self.data_collector.dataframes[set+'_middle'].append(middle_line)
        self.data_collector.dataframes[set] = self.data_collector.dataframes[set+'_middle']
        del self.data_collector.dataframes[set+'_middle']

    def _train_w2v(self):
        model, sentance = self.handler.train_w2v(embedding_dim=config.embedding_dim, min_count=self.min_count, num_works=5)
        self.vocab.build(sentance, model, 1)
        self.handler.vocab = self.vocab
        pk.dump(self.vocab, open(os.path.join(config.vocab_folder, config.vocab_name), 'wb'))

        del self.vocab

    def _process2processed(self):
        for set in self.data_collector.dataframes.keys():
            self._process2processed_set(set)

    def _process2processed_set(self, set='train'):
        assert set in ['train', 'test', 'dev']
        func = self.handler.line_to_processed
        self.data_collector.dataframes[set + '_processed'] = []
        for line in tqdm(self.data_collector.dataframes[set], desc=set + '_middle'):
            middle_line = {}
            middle_line['content'], middle_line['title'], middle_line['id'] = func(line['content']), func(line['title']), line['id']
            self.data_collector.dataframes[set + '_processed'].append(middle_line)
        self.data_collector.dataframes[set] = self.data_collector.dataframes[set + '_processed']
        del self.data_collector.dataframes[set + '_processed']

    def _save(self):
        for set in self.data_collector.dataframes.keys():
            self._save_file(set)

    def _save_file(self, set):
        if os.path.exists(os.path.join(config.byte_root, set)):
            shutil.rmtree(os.path.join(config.byte_root, set))
        os.mkdir(os.path.join(config.byte_root, set))
        for row in tqdm(self.data_collector.dataframes[set]):
            json.dump(row, open(os.path.join(config.byte_root, set, str(row['id']) + '.json'), 'w'), ensure_ascii=False)

    def process(self):
        self._process2middle()
        self._train_w2v()
        self._process2processed()
        self._save()


class Checker(object):
    def __init__(self, data_collector, data_pipe):
        self.data_collector = data_collector
        self.data_pipe = data_pipe
        self.files = os.listdir(os.path.join(config.byte_root, 'train'))

    def check(self, set='train'):
        for i in self.data_collector.dataframes[set]:
            print(f'{set}:')
            content = i['content']
            print(f'raw:{content}')
            print('----------------')
            processed = ' '.join(self.data_pipe.line_to_middle(content))
            print(f'middle:{processed}')
            print('=========================================')
            n = input('next:...')


def sample():
    data_collector = DataCollector(sample=True)
    data_collector.collect_files()
    data_processor = DataProcessor(data_collector=data_collector, min_count=1)
    data_processor.process()


def process():
    data_collector = DataCollector(sample=False)
    data_collector.collect_files()
    data_processor = DataProcessor(data_collector=data_collector, min_count=200)
    data_processor.process()


def check():
    data_collector = DataCollector(sample=True)
    data_collector.collect_files()
    data_pipe = ByteDataPipe()
    checker = Checker(data_collector, data_pipe)
    checker.check()


if __name__ == '__main__':
    fire.Fire()

