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
import pickle
sys.path.append('..')
sys.path.append('../datasets')
import config
from collections import OrderedDict, defaultdict
from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess
import data_propbankbr as br


SCHEMA_PATH = '../{:}gs.yaml'.format(config.SCHEMA_DIR)

class _EncoderIterator(object):
    def __init__(self, low, high, decoder_fn):
        self.low = low
        self.high = high
        self.decoder_fn = decoder_fn
        self.current = self.low

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1, self.decoder_fn(self.current - 1)


class PropbankEncoder(object):
    '''
        PropbankEncoder mantains an indexed representation of the database
        There are numerical types, str types and categorical types.
        The return format will depend on requested encoding

        Encondings:
        CAT     .:  tokens and categorical values will be returned
        EMB     .:  tokens will be embeded into the language model,
                    categorical values will be onehot encoded
        HOT     .:  tokens and categorical values will be onehot encoded
        IDX     .:

    '''

    def __init__(self):
        # Pickled variables must live within __init__()
        self.lexicon = set([])
        self.lex2tok = {}
        self.tok2idx = {}
        self.idx2tok = {}
        self.embeddings = np.array([])
        self.embeddings_model = ''
        self.embeddings_sz = 0

        self.onehot = defaultdict(OrderedDict)
        self.hotone = defaultdict(OrderedDict)
        self.db = defaultdict(OrderedDict)
        self.encodings = ('CAT', 'EMB', 'HOT', 'IDX')
        self.schema_d = {}
        self.columns_mapper = {}
        self.columns = set([])

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

    def iterator(self, ds_type, filter_columns=['P', 'FORM', 'ARG'], encoding='EMB'):
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



            interval = [idx
                        for idx, p in self.db['P'].items()
                        if p > lb and p <= ub]
            low = min(interval)
            high = max(interval)

        if filter_columns:
            for col in filter_columns:
                if not col in self.db:
                    errmessage = 'column {:} not in db columns {:}'.format(col, self.db)
                    raise ValueError(errmessage)
        else:
            filter_columns = list(self.db.keys())

        if not encoding in self.encodings:
            _errmessage = 'encoding {:} not in {:}'.format(encoding, self.encodings)
            raise ValueError(_errmessage)
        else:
            fn = lambda x: self._decode_with_idx(x, filter_columns, encoding)

        return _EncoderIterator(low, high, fn)

    def column_dimensions(self, column, encoding):
        if encoding in ('IDX', 'CAT'):
            return 1

        base_col = self.columns_mapper.get(column, None)
        if not base_col:
            return 1

        col_type = self.schema_d[base_col]['type']
        if not col_type:
            return 1

        if encoding in ('HOT'):
            if col_type in ('str'):
                return len(self.lexicon)
            elif col_type in ('choice'):
                return len(self.onehot[base_col])
            else:
                return 1

        if encoding in ('EMB'):
            if col_type in ('str'):
                return self.embeddings_sz
            elif col_type in ('choice'):
                return len(self.onehot[base_col])
            else:
                return 1

    def columns_dimensions(self, encoding):
        return {
            col: self.column_dimensions(col, encoding) for col in self.columns
        }

    def column(self, ds_type, column, encoding):
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

        return {
                x:self._decode_with_idx(x,[column], encoding)
                for x, p in self.db['P'].items()
                if p > lb and p <= ub
                }

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

        # embeddings_shape = (len(tokens) + 1, self.embeddings_sz) # unk token
        self.tok2idx = {'unk': 0}
        self.idx2tok = {0: 'unk'}
        # self.embeddings = np.zeros(embeddings_shape, dtype=np.float32)
        self.embeddings = []
        self.embeddings.append(word2vec['unk'].tolist())

        i = 1
        for tok in sorted(list(tokens)):
            self.tok2idx[tok] = i
            self.idx2tok[i] = tok
            self.embeddings.append(word2vec[tok].tolist())
            i += 1

    def _process_onehot(self):
        for col in self.schema_d:
            if self.schema_d[col]['type'] in ('choice'):
                _keys = ['unk'] + sorted(self.schema_d[col]['domain'])
                _values = list(range(len(_keys)))
                d = dict(zip(_keys, _values))
                self.onehot[col] = OrderedDict(sorted(d.items(), key=lambda x: x[0]))

                d = dict(zip(_values, _keys))
                self.hotone[col] = OrderedDict(sorted(d.items(), key=lambda x: x[0]))

    def _process_db(self, dbpt_d, dbname):
        self.dbname = dbname
        # Computes a dictionary that maps one column to a base column
        self.columns = self.columns.union(dbpt_d.keys())
        self.columns_mapper = {col: re.sub(r'[\+|\-|\d|]|(_CTX_P)', '', col)
                               for col in self.columns}

        for col in list(self.columns):
            base_col = self.columns_mapper[col]
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
                        idx: self.tok2idx[self.lex2tok.get(word,'unk')] for idx, word in dbpt_d[col].items()
                    })
                else:
                    #Boolean values, numerical values come here
                    self.db[col] = OrderedDict(dbpt_d[col])

    def tensor2column(self, tensor_index, tensor_values, times, column):
        '''
            Converts a zero padded tensor to a dict
        
            Tensors must have the following shape [DATABASE_SIZE, MAX_TIME] with
                zeros if for t=0,....,MAX_TIME t>times[i] for i=0...DATABASE_SIZE 

        args:
            tensor_index  .: with db index

            tensor_values .: with db int values representations

            times  .: list<int> [DATABASE_SIZE] holding the times for each proposition

            column .: str           db column name

        returns:
            column .: dict<int, str> keys in db_index, values in columns or values in targets
        '''
        if not(column) in self.db:
            buff= '{:} must be in {:}'.format(column, self.db)
            raise KeyError(buff)
        
        # import code; code.interact(local=dict(globals(), **locals()))
        index=[item  for i, sublist in enumerate(tensor_index.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        values = [self.hotone[column][item]
            for i, sublist in enumerate(tensor_values.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        return dict(zip(index, values))

    def t2arg(self, T):
        '''
            Converts column T into ARG

            args:
                T .: dict<int, str> keys in db_index, values: prediction label

            args:
                ARG .: dict<int, str> keys in db_index, values in target label
        '''        
        propositions= {idx: self.db['P'][idx] for idx in T}

        ARG = br.propbankbr_t2arg(propositions.values(), T.values())

        return dict(zip(T.keys(), ARG))

    def _decode_with_idx(self, idx, columns, encoding):
        d = OrderedDict()
        for col in columns:
            val = self.db[col].get(idx, 0)
            d[col] = self._decode_with_value(val, col, encoding)

        if len(columns) == 1:
            return d[col]
        else:
            return d

    def _decode_with_value(self, x, column, encoding):
        base_col = self.columns_mapper.get(column, None)

        if base_col and base_col in self.schema_d:
            col_type = self.schema_d[base_col]['type']

        if encoding in ('IDX') or not base_col or col_type in ('int', 'bool'):
            return x

        elif encoding in ('CAT'):
            if col_type in ('choice'):
                return self.hotone[base_col][x]
            elif col_type in ('str'):
                return self.idx2tok[x]
            else:
                return x

        elif encoding in ('EMB'):
            if col_type in ('str'):
                return self.embeddings[x]
            elif col_type in ('choice'):
                sz = self.column_dimensions(column, encoding)

                return [1 if i == x else 0 for i in range(sz)]
            else:
                return x

        elif encoding in ('HOT'):
            sz = self.column_dimensions(column, encoding)

            return [1 if i == x else 0 for i in range(sz)]
        else:
            raise Exception('Unhandled exception')


if __name__ == '__main__':
    dfgs = pd.read_csv('../datasets/csvs/gs.csv', index_col=None)
    dfgs.set_index('INDEX', inplace=True)
    dfgs['INDEX'] = dfgs.index.tolist()


    column_files = [
        '../datasets/csvs/column_predmarker/predicate_marker.csv',
        '../datasets/csvs/column_shifts_ctx_p/form.csv',
        '../datasets/csvs/column_t/t.csv'
    ]

    for col_f in column_files:
        _df = pd.read_csv(col_f, index_col=0, encoding='utf-8')
        dfgs = pd.concat((dfgs, _df), axis=1)

    propbank_encoder = PropbankEncoder().define(dfgs.to_dict())
    print(propbank_encoder.db.keys())
    propbank_encoder.persist('../datasets/binaries/', filename='deep')
    # propbank_encoder = PropbankEncoder.recover('../datasets/binaries/deep.pickle')
    # filter_columns = ['P', 'GPOS', 'FORM']
    # for t, d in propbank_encoder.iterator('test', filter_columns=filter_columns, encoding='EMB'):
    #     print('t:{:}\tP:{:}\tGPOS:{:}\tFORM:{:}'.format(t, d['P'], d['GPOS'], d['FORM']))

    # for t, d in propbank_encoder.iterator('train', filter_columns=filter_columns, encoding='CAT'):
    #     print('t:{:}\tP:{:}\tGPOS:{:}\tFORM:{:}'.format(t, d['P'], d['GPOS'], d['FORM']))

    # for t, d in propbank_encoder.iterator('valid', filter_columns=filter_columns, encoding='HOT'):
    #     print('t:{:}\tP:{:}\tGPOS:{:}'.format(t, d['P'], d['GPOS']))
