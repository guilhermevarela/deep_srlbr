'''
    Created on Apr 20, 2018
        @author: Varela

    PropbankEncoder wrapps a set of columns indexing them
'''
import re
import yaml
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import config
from collections import OrderedDict, defaultdict
from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess



SCHEMA_PATH = '../{:}gs.yaml'.format(config.SCHEMA_DIR)
# EMBEDDINGS_PATH = '../{:'.format(config.LANGUAGE_MODEL_DIR)

class _EncoderIterator(object):
    def __init__(self, low, high, dbparser_fn):
        self.low = low
        self.high = high
        self.dbparser_fn = dbparser_fn
        self.current = self.low


class PropbankEncoder(object):
    '''
        PropbankEncoder mantains an indexed representation of the database
        There are numerical types, str types and categorical types. The return
        format will depend on the desired format. 

        Encondings:
        CAT     .:  tokens and categorical values will be returned                   
        EMB     .:  tokens will be embeded into the language model,
                    categorical values will be onehot encoded
        HOT     .:  tokens and categorical values will be onehot encoded
        IDX     .:

    '''
    lexicon = set([])
    lex2tok = {}
    tok2idx = {}
    idx2tok = {}
    embeddings = np.array([])
    embeddings_model = ''
    embeddings_sz = 0

    onehot = defaultdict(OrderedDict)
    hotone = defaultdict(OrderedDict)
    db = defaultdict(OrderedDict)
    encodings = ('CAT', 'EMB', 'HOT', 'IDX')
    schema_d = {}
    columns_mapper_d = {}
    columns = set(['INDEX'])

    def __init__(self):
        # loads schema so that indexes will be the same
        with open(SCHEMA_PATH, mode='r') as f:
            self.schema_d = yaml.load(f)

        # fills lexicon
        for col in self.schema_d:
            if self.schema_d[col]['type'] == 'str':
                self.lexicon = self.lexicon.union(self.schema_d[col]['domain'])

        # SOLVES CATEGORICAL        
        self._process_onehot()

    def define(self, dbpt_d, language_model='glove_s50', dbname='dbpt', verbose=True):

        # RUNS EMBEDDINGS
        self._process_embeddings(language_model, verbose)

        # COMPUTES data dict
        self._process_db(dbpt_d, dbname)

        # GOOD FOR CHAINING
        return self

    @classmethod
    def recover(cls, file_path):
        '''
        Returns a copy from the instanced saved @file_path
        args:
            file_path .: string a full path to a serialized dump of propbank object

        returns:
             object   .: propbank object instanced saved at file_path
        '''
        with open(file_path, 'rb') as f:
            propbank_instance = pickle.load(f)
        return propbank_instance

    def persist(self, file_dir, filename=''):
        '''
        Serializes this object in pickle format @ file_dir+filename
        args:
            file_dir    .: string representing the directory
            filename  .: string (optional) filename
    '''
        if not(filename):
            _moniker = (file_dir, self.dbname, self.embeddings_model, self.embeddings_sz)
            filename = '{:}{:}_{:}_s{:}.pickle'.format(*_moniker)
        else:
            filename = '{:}{:}.pickle'.format(file_dir, filename)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def iterator(self, ds_type, columns, encoding='EMB'):
        if not(ds_type in ['train', 'valid', 'test']):
            errmessage = 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
            raise ValueError(errmessage)
        else:
            if ds_type in ['train']:
                lb = 0
                ub = config.DATASET_TRAIN_SIZE
            elif ds_type in ['valid']:
                lb = config.DATASET_TRAIN_SIZE
                ub = config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE
            else:
                lb = config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE
                ub = config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE + config.DATASET_TEST_SIZE

        for col in columns:
            if not col in self.db:
                errmessage = 'column {:} not in db columns {:}'.format(col, self.columns)
                raise ValueError(errmessage)


        if not encoding in self.encodings:
            _errmessage = 'encoding {:} not in {:}'.format(encoding, self.encodings)
            raise ValueError(_errmessage)
        else:
            fn = lambda x: self._decode(x, encoding)

    def _process_embeddings(self, language_model, verbose):
        # computes embeddings
        word2vec = fetch_word2vec(language_model, verbose=verbose)
        self.embeddings_model = language_model.split('_s')[0]
        self.embeddings_sz = int(language_model.split('_s')[1])

        self.lex2tok = preprocess(list(self.lexicon), word2vec)
        if verbose:
            words = self.lex2tok.keys()
            tokens = set(self.lex2tok.values())
            print('# UNIQUE TOKENIZED {:}, # EMBEDDED TOKENS {:}'.format(len(words), len(tokens)))

        embeddings_shape = (len(tokens) + 1, self.embeddings_sz) # unk token
        self.tok2idx = {'unk': 0}
        self.idx2tok = {0: 'unk'}
        self.embeddings = np.zeros(embeddings_shape, dtype=np.float32)
        self.embeddings[0] = word2vec['unk']

        i = 1
        for tok in list(tokens):
            self.tok2idx[tok] = i
            self.idx2tok[i] = tok
            self.embeddings[i] = word2vec[tok]
            i += 1

    def _process_onehot(self):
        for col in self.schema_d:
            if self.schema_d[col]['type'] in ('choice'):
                _keys = list(set(self.schema_d[col]['domain']))
                _values = list(range(len(_keys)))
                self.onehot[col] = OrderedDict(dict(zip(_keys, _values)))
                self.hotone[col] = OrderedDict(dict(zip(_values, _keys)))

    def _process_db(self, dbpt_d, dbname):
        self.dbname = dbname
        # Computes a dictionary that maps one column to a base column
        self.columns = self.columns.union(dbpt_d.keys())
        self.columns_mapper = {col: re.sub(r'[\+|\-|\d|]|(_CTX_P)', '', col)
                               for col in self.columns}

        for col in list(self.columns):
            base_col = self.columns_mapper[col]
            try:
                if col in ('INDEX') or not base_col in self.schema_d:
                    #Boolean values, numerical values come here
                    self.db[col] = OrderedDict(dbpt_d[col])
                else:                
                    if self.schema_d[base_col]['type'] in ('choice'):
                        self.db[col] = OrderedDict({
                            idx: self.onehot[base_col][category] for idx, category in dbpt_d[col].items()
                        })

                    elif self.schema_d[base_col]['type'] in ('str'):
                        self.db[col] = OrderedDict({
                            idx: self.tok2idx[self.lex2tok[word]] for idx, word in dbpt_d[col].items()
                        })
                    else:                    
                        #Boolean values, numerical values come here
                        self.db[col] = OrderedDict(dbpt_d[col])
            except KeyError:
                import code; code.interact(local=dict(globals(), **locals()))

    def _decode(self, x, encoding):


if __name__ == '__main__':    
    dfgs = pd.read_csv('../datasets/csvs/gs.csv')
    propbank_encoder = PropbankEncoder().define(dfgs.to_dict())
    import code; code.interact(local=dict(globals(), **locals()))
    propbank_encoder.schema_d
