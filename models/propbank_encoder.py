'''
    Created on Apr 20, 2018
        @author: Varela

    PropbankEncoder wraps a set of columns storing an integer indexed representation
        of the original dataset atributes. Provides column information
'''
import re
import yaml
import numpy as np
import pandas as pd
import sys, os
import pickle
import glob

root_path = re.sub('/models', '', sys.path[0])
sys.path.append(root_path)

import config
from collections import OrderedDict, defaultdict
from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess
# import data_propbankbr as br





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
    '''Mantains a representation of the database

        PropbankEncoder mantains an indexed representation of the database
        There are numerical types, str types and categorical types.
        The return format will depend on requested encoding

        Encondings:
        CAT     .:  tokens and categorical values will be returned
        EMB     .:  tokens will be embeded into the language model,
                    categorical values will be onehot encoded
        HOT     .:  tokens and categorical values will be onehot encoded
        IDX     .:  raw indexes

    '''


    def __init__(self, db_dict, schema_dict, language_model='glove_s50', dbname='dbpt', verbose=True):
        '''Initializer processes raw dataset extracting the numerical representations

        Provides a wrapper around the database

        Arguments:
            db_dict {dict<str, dict<int, object>>} -- nested dictionary representing
                database, outer_keys: columns, inner_keys: indexes.
            schema_dict {dict<str, dict><>} -- dictionary representaiton from schema
                database, outer_keys: columns, inner_keys: meta data


        Keyword Arguments:
            language_model {str} -- embeddings to be used (default: {'glove_s50'})
            dbname {str} -- default dbname (default: {'dbpt'})
            verbose {bool} -- display operations (default: {True})
        '''

        self.lex2idx = {}
        self.idx2lex = {}
        self.tokens = set({}) # words after embedding model
        self.words = set({})  # raw words that come within datasets
        self.lex2tok = defaultdict(OrderedDict)
        self.tok2idx = defaultdict(OrderedDict)
        self.idx2tok = {}
        self.embeddings = np.array([])
        self.embeddings_model = ''
        self.embeddings_sz = 0


        self.db = defaultdict(OrderedDict)
        self.encodings = ('CAT', 'EMB', 'HOT', 'IDX')
        self.schema_dict = {}
        self.columns_mapper = {}
        self.columns_config = {}
        self.columns = set([])

        self._process(db_dict, schema_dict, dbname, language_model, verbose)


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

    def iterator(self, ds_type, filter_columns=['P', 'T'], encoding='EMB'):
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



            interval = [idx for idx, p in self.db['P'].items()
                        if p > lb and p <= ub]
            low = min(interval)
            high = max(interval)

        if filter_columns:
            for col in filter_columns:
                if not col in self.db:
                    errmessage = 'column {:} not in db columns {:}'.format(col, self.db.keys())
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
        if not encoding in self.encodings:
            raise ValueError('Supported encondings are {:} got {:}.'.format(self.encodings, encoding))

        if encoding in ('IDX', 'CAT'):
            return 1

        colconfig = self.columns_config.get(column, None)
        if not colconfig:
            return 1

        if encoding in ('HOT'):
            if colconfig['type'] in ('text', 'choice'):
                return colconfig['dims']
            else:
                return 1

        if encoding in ('EMB'):
            if colconfig['type'] in ('text'):
                return self.embeddings_sz
            elif colconfig['type'] in ('choice'):
                return colconfig['dims']
            else:
                return 1

    def columns_dimensions(self, encoding):
        return {
            col: self.column_dimensions(col, encoding) for col in self.columns
        }

    def column(self, ds_type, column, encoding):
        if not(ds_type in ['train', 'valid', 'test']):
            _msg = 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''
            _msg = _msg.format(ds_type)
            raise ValueError(_msg)
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


    def _process(self, db_dict, schema_dict, dbname, language_model, verbose):
        # Defines domains for each column, defines tokens for text columns
        self._process_lexicons(db_dict, schema_dict)

        # MAPS textual columns to the language model
        self._process_embeddings(language_model, verbose)

        # Builds db dictionary
        self._process_db(db_dict, dbname)

    def _process_embeddings(self, language_model, verbose):
        # computes embeddings
        word2vec = fetch_word2vec(language_model, verbose=verbose)
        self.embeddings_model = language_model.split('_s')[0]
        self.embeddings_sz = int(language_model.split('_s')[1])


        self.lex2tok = preprocess(list(self.words), word2vec)
        if verbose:
            words = self.lex2tok.keys()
            self.tokens = set(self.lex2tok.values())
            print('# UNIQUE TOKENIZED {:}, # EMBEDDED TOKENS {:}'.format(len(words), len(self.tokens)))


        self.tok2idx = {'unk': 0}
        self.idx2tok = {0: 'unk'}
        self.embeddings = []
        self.embeddings.append(word2vec['unk'].tolist())

        i = 1
        for tok in sorted(list(self.tokens)):
            if not tok in self.tok2idx:
                self.tok2idx[tok] = i
                self.idx2tok[i] = tok
                self.embeddings.append(word2vec[tok].tolist())
                i += 1

    def _process_db(self, db_dict, dbname):
        self.dbname = dbname

        for col in list(self.columns):
            base_col = self.columns_mapper[col]
            colconfig = self.columns_config.get(col, None)
            if col in ('INDEX',) or not colconfig:
                #Boolean values, numerical values come here
                if col in ('INDEX',):
                    self.db[col] = OrderedDict(
                        dict(zip(db_dict['FORM'].keys(),
                                 db_dict['FORM'].keys()
                    )))
                else:
                    self.db[col] = OrderedDict(db_dict[col])
            elif colconfig['type'] in ('choice', 'text'):
                self.db[col] = OrderedDict({
                    idx: self.lex2idx[col].get(word, 0) for idx, word in db_dict[col].items() 
                })

            else:
                #Boolean values, numerical values come here
                self.db[col] = OrderedDict(db_dict[col])


    def _process_lexicons(self, db_dict, schema_dict):
        '''
            Define columns and metadata about columns
        '''
        # Computes a dictionary that maps one column to a base column

        self.columns = self.columns.union(db_dict.keys()).union(set(['INDEX']))
        self.columns_mapper = {col: self._subcol(col)
                               for col in self.columns}

        # Creates descriptors about the data
        dflt_dict =  {'name':'dflt', 'category':'feature', 'type':'int', 'dims': None}
        for col in list(self.columns):
            base_col = self.columns_mapper[col]

            # Defines metadata

            if base_col in schema_dict:
                colitem = schema_dict[base_col]
                domain = colitem.get('domain', None) # numerical values                

                config_dict = {key: colitem.get(key, col) for key in ['name', 'category', 'type']}

                config_dict['dims'] = len(domain) if domain else None
            else:
                config_dict = dflt_dict
                config_dict['name'] = col
                domain = None


            # Lexicon for each column unk is not present
            if config_dict['type'] in ('text') or \
                (config_dict['category'] in ('feature') and config_dict['type'] in ('choice')):
                # features might be absent ( in case of leading and lagging )
                sorted(domain).insert(0, 'unk')

                if config_dict['type'] in ('text'):
                    self.words = self.words.union(set(domain))

            if domain:
                self.lex2idx[col] = dict(zip(domain, range(len(domain))))
                self.idx2lex[col] = dict(zip(range(len(domain)), domain))
                config_dict['dims'] = len(domain)

            self.columns_config[col] = config_dict

    def _subcol(self, col):
        re_ctxp = r'(_CTX_P)|(_\d)|[\+|\-|\d|]'
        re_repl = r'(_CHILD)|(_PARENT)|(_GRAND_PARENT)'

        bcol = re.sub(re_ctxp, '', col)
        bcol = re.sub(re_repl, '', bcol)
        return bcol

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
        colconfig = self.columns_config[column]
        if encoding in ('IDX') or colconfig['type'] in ('int', 'bool'):
            return x

        elif encoding in ('CAT'):
            if colconfig['type'] in ('choice', 'text'):
                return self.idx2lex[column][x]
            else:
                return x

        elif encoding in ('EMB'):
            if colconfig['type'] in ('text'):
                word = self.idx2lex[column][x]
                token = self.lex2tok[word]
                return self.embeddings[self.tok2idx[token]]

            elif colconfig['type'] in ('choice'):
                sz = self.column_dimensions(column, encoding)
                return [1 if i == x else 0 for i in range(sz)]

            else:
                return x

        elif encoding in ('HOT'):
            sz = self.column_dimensions(column, encoding)

            return [1 if i == x else 0 for i in range(sz)]
        else:
            raise Exception('Unhandled exception')
