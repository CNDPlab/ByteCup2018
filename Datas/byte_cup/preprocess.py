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


args = DefaultConfigs()


class DataCollector(object):
    def __init__(self, sample):
        self.data_root = 'Datas/byte_cup/raw/'
        self.prefix = 'bytecup.corpus.train'
        self.files = [file for file in os.listdir(self.data_root) if self.prefix in file]
        self.collected_file = 'all_data.json'
        self.get_keys()
        self.sample = sample
        self.columns = {}
        self.dataframes = {'train':None, 'dev':None, 'test':None}

    def get_keys(self):
        with open(os.path.join(self.data_root, self.files[0])) as reader:
            for line in reader:
                self.keys = json.loads(line).keys()
                break
        self.columns = {i: [] for i in self.keys}

    def collect_files(self):
        if not os.path.exists(os.path.join(self.data_root, self.collected_file)):
            for file in tqdm(self.files, desc='files'):
                self.collect_lines(file)
            self.df = pd.DataFrame.from_dict(self.columns)
            self.df.to_json(os.path.join(self.data_root, self.collected_file))
        else:
            self.df = pd.read_json(os.path.join(self.data_root, self.collected_file))
        self.split_train_test()

    def collect_lines(self, file):
        if self.sample:
            with open(os.path.join(self.data_root, file)) as reader:
                for index, line in tqdm(enumerate(reader), desc='lines'):
                    if index > 200:
                        break
                    for key in self.keys:
                        self.columns[key].append(json.loads(line)[key])
        else:
            with open(os.path.join(self.data_root, file)) as reader:
                for index, line in tqdm(enumerate(reader), desc='lines'):
                    for key in self.keys:
                        self.columns[key].append(json.loads(line)[key])

    def split_train_test(self):
        self.dataframes['train'], self.dataframes['dev'] = train_test_split(self.df, test_size = 0.01, random_state=1)
        self.dataframes['dev'], self.dataframes['test'] = train_test_split(self.dataframes['dev'], test_size=0.5, random_state=1)
        del self.df


class DataProcessor(object):
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.handler = ByteDataPipe()
        self.vocab = Vocab()

    def _process2middle_train(self):
        tqdm.pandas('train')
        self.data_collector.dataframes['train']['middle_content'] = \
            self.data_collector.dataframes['train']['content'].progress_apply(self.handler.line_to_middle_train)
        self.data_collector.dataframes['train']['middle_title'] = \
            self.data_collector.dataframes['train']['title'].progress_apply(self.handler.line_to_middle_train)

    def _process2middle_dev_test(self):
        tqdm.pandas('dev')
        self.data_collector.dataframes['dev']['middle_content'] = \
            self.data_collector.dataframes['dev']['content'].progress_apply(self.handler.line_to_middle_dev)
        self.data_collector.dataframes['dev']['middle_title'] = \
            self.data_collector.dataframes['dev']['title'].progress_apply(self.handler.line_to_middle_dev)

        tqdm.pandas('test')
        self.data_collector.dataframes['test']['middle_content'] = \
            self.data_collector.dataframes['test']['content'].progress_apply(self.handler.line_to_middle_dev)
        self.data_collector.dataframes['test']['middle_title'] = \
            self.data_collector.dataframes['test']['title'].progress_apply(self.handler.line_to_middle_dev)

    def _train_w2v(self):
        model, sentance = self.handler.train_w2v(args.embedding_dim, 1, 1)
        self.vocab.build(sentance, model, 1)
        self.handler.vocab = self.vocab
        del self.vocab

    def _process2processed(self):
        for set in self.data_collector.dataframes.keys():
            self._process2processed_set(set)

    def _process2processed_set(self, set='train'):
        tqdm.pandas('processed')
        self.data_collector.dataframes[set]['processed_content'] = \
            self.data_collector.dataframes['processed_content'].progress_apply(self.handler.line_to_processed)
        self.data_collector.dataframes[set]['processed_title'] = \
            self.data_collector.dataframes['processed_title'].progress_apply(self.handler.line_to_processed)

    def _save(self):
        for set in self.data_collector.dataframes.keys():
            self._save_file(set)

    def _save_file(self, set):
        if os.path.exists(os.path.join(args.byte_root, set)):
            shutil.rmtree(os.path.join(args.byte_root, set))
        os.mkdir(os.path.join(args.byte_root, set))
        for index, row in tqdm(enumerate(self.data_collector.dataframes[set].iterrows())):
            row[1].to_json(os.path.join(args.byte_root, set, str(index), '.json'))

    def process(self):
        self._process2middle_train()
        self._process2middle_dev_test()
        self._train_w2v()
        self._process2processed()
        self._save()

class Checker(object):
    def __init__(self):
        self.data_root = args.byte_processed
        self.files = os.listdir(self.data_root)

    def check(self):
        for i in self.files:
            data = pd.read_json(os.path.join(self.data_root, i))
            print(f'content:')
            print(f'raw:{data.content}')
            print(f'middle:{data.middle_content}')
            print(f'title')
            print(f'raw:{data.title}')
            print(f'middle:{data.middle_title}')



def main(task):
    if task == 'test':
        data_collector = DataCollector(True)
        data_collector.collect_files()
        data_processor = DataProcessor(data_collector)
        data_processor.process()

    if task == 'process':
        data_collector = DataCollector(False)
        data_collector.collect_files()
        data_processor = DataProcessor(data_collector)
        data_processor.process()

    if task == 'check':
        checker = Checker()
        checker.check()


if __name__ == '__main__':
    fire.Fire()






