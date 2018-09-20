import warnings
import os



class DefaultConfigs():
    data_root = 'Datas/'

    byte_root = os.path.join(data_root, 'byte_cup')
    byte_raw = os.path.join(byte_root, 'raw')

    sog_root = os.path.join(data_root, 'sog')
    sog_raw = os.path.join(sog_root, 'raw')


    load_from_exp = None

    batch_zie = 64
    hidden_size = 64
    embedding_dim = 256
    model_name = 'Transformer'
    vocab_folder = 'Predictor/Utils/Vocab/'
    vocab_name = 'vocab_byte.pkl'
    ckpt_root = 'ckpt/'
    eos_id = None
    sos_id = None
    warm_up_step = 4000

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print('__', k, getattr(self, k))

