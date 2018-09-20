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
# import cps import fetch_word2vec, preprocess
# import models.utils as cps
import utils
from utils import corpus as cps


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
        CAT     .:  (categorical) text column will be a string,
                    categorical column will be the category
                    -- use case: display

        EMB     .:  (embedded) tokens will be embeded into the language model,
                    categorical values will be one-hot encoded
                    -- use case: tensorflow for fixed embeddings

        HOT     .:  (one-hot) text will be one-hot encoded,
                    categorical values will be one-hot encoded
                    -- use case: Maximal Margin Separators

        IDX     .:  (indexed) text and categorical values are indexed
                    -- use case: dense representation and debugging

        MIX     .:  (mix) text will return  word index
                    categorical will be one-hot encoded
                    -- use case: trainable embeddings


    '''
    def __init__(self, db_dict, schema_dict,
                 language_model='glove_s50', dbname='dbpt', filter_br=True,
                 version='1.0', verbose=True):
        '''Initializer processes raw dataset extracting the numerical representations

        Provides a wrapper around the database

        Arguments:
            db_dict {dict<str, dict<int, object>>} -- 
                nested dictionary representing
                database, outer_keys: columns, inner_keys: indexes.
            schema_dict {dict<str, dict><>} -- 
                dictionary representaiton from schema
                database, outer_keys: columns, inner_keys: meta data

        Keyword Arguments:
            language_model {str} -- embeddings to be used
                (default: {'glove_s50'})
            dbname {str} -- default dbname (default: {'dbpt'})
            filter_br {bool} -- If True replaces
                non continous arguments with referenced
                arguemnts (default: {True})
            verbose {bool} -- display operations (default: {True})
        '''
        self.filter_br = filter_br
        self.lex2idx = {}
        self.idx2lex = {}
        self.tokens = []  # words after embedding model
        self.words = []   # raw words that come within datasets
        self.lex2tok = defaultdict(OrderedDict)
        self.embeddings = np.array([])
        self.embeddings_model = ''
        self.embeddings_sz = 0

        self.db = defaultdict(OrderedDict)
        self.encodings = ('CAT', 'EMB', 'HOT', 'IDX', 'MIX')
        self.schema_dict = {}
        self.columns_mapper = {}
        self.columns_config = {}
        self.columns = set([])
        self.version = version
        self._initialize(db_dict, schema_dict, dbname, language_model, verbose)

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

            args_list = [file_dir, self.dbname]
            args_list += [self.embeddings_model, self.embeddings_sz]
            filename = '{:}{:}_{:}_s{:}.pickle'.format(*args_list)

        else:
            filename = '{:}{:}.pickle'.format(file_dir, filename)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def iterator(self, ds_type, filter_columns=['P', 'T'], encoding='EMB'):
        lb, ub = utils.get_db_bounds(ds_type, version=self.version)

        interval = [idx for idx, p in self.db['P'].items()
                    if p >= lb and p < ub]

        low = min(interval)
        high = max(interval)

        if filter_columns:
            for col in filter_columns:
                if col not in self.db:
                    err = 'column {:} not in db columns {:}'
                    err = err.format(col, self.db.keys())
                    raise ValueError(err)
        else:
            filter_columns = list(self.db.keys())

        if encoding not in self.encodings:
            err = 'encoding {:} not in {:}'
            err = err.format(encoding, self.encodings)
            raise ValueError(err)

        else:
            def fn(x):
                return self._decode_index(x, filter_columns, encoding)

        return _EncoderIterator(low, high, fn)

    def column_dimensions(self, column, encoding):
        if encoding not in self.encodings:
            err = 'Supported encondings are {:} got {:}.'
            err = err.format(self.encodings, encoding)
            raise ValueError(err)

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

        if encoding in ('MIX'):

            if colconfig['type'] in ('choice'):
                return colconfig['dims']
            else:
                return 1

    def columns_dimensions(self, encoding):
        return {
            col: self.column_dimensions(col, encoding) for col in self.columns
        }

    def column(self, ds_type, column, encoding):
        lb, ub = utils.get_db_bounds(ds_type)

        return {x:self._decode_index(x,[column], encoding)
                for x, p in self.db['P'].items()
                if p > lb and p <= ub}


    def _initialize(self, db_dict, schema_dict, dbname, language_model, verbose):
        # Defines domains for each column, defines tokens for text columns
        self._initialize_lexicons(db_dict, schema_dict)

        # MAPS textual columns to the language model
        self._initialize_embeddings(language_model, verbose)

        # Builds db dictionary
        self._initialize_db(db_dict, dbname)

    def _initialize_embeddings(self, language_model, verbose):
        # computes embeddings
        word2vec = cps.fetch_word2vec(language_model, verbose=verbose)
        self.embeddings_model = language_model.split('_s')[0]
        self.embeddings_sz = int(language_model.split('_s')[1])

        self.lex2tok = cps.preprocess(self.words, word2vec, verbose=verbose)
        if verbose:
            args_ = (len(self.words), len(set(self.lex2tok.values())))
            print('# unique words {:}, tokens {:}'.format(*args_))

        self.tokens = []
        self.embeddings = []
        self.tok2idx = {}
        self.idx2tok = {}
        i = 0
        for word, idx in self.word2idx.items():

            token_ = self.lex2tok[word]
            if token_ not in self.tok2idx:
                if '_' not in token_:
                    embs_ = word2vec[token_]
                else:
                    # This token is composed of multiple tokens
                    # Make mean from the words
                    for j, tok_ in enumerate(token_.split('_')):
                        if j == 0:
                            embs_ = word2vec[tok_]
                        else:
                            embs_ = word2vec[tok_] + embs_

                    embs_ = embs_ / (j + 1)

                self.embeddings.append(embs_.tolist())
                self.tok2idx[token_] = i
                self.idx2tok[i] = token_
                self.tokens.append(token_)
                i += 1

    def _initialize_db(self, db_dict, dbname):
        self.dbname = dbname
        for col in list(self.columns):
            colconfig = self.columns_config.get(col, None)

            if col in ('INDEX',) or not colconfig:  # numeric / bool values

                if col in ('INDEX',):
                    keys_ = db_dict['FORM'].keys()
                    self.db[col] = OrderedDict(
                        dict(zip(keys_, keys_))
                    )
                else:
                    self.db[col] = OrderedDict(db_dict[col])

            elif colconfig['type'] in ('choice'):
                if colconfig['category'] == 'target' and self.filter_br:
                        self.db[col] = OrderedDict({
                            idx: self.lex2idx[col].get(re.sub('C-', '', word), 0)
                            for idx, word in db_dict[col].items()
                        })
                else:
                    self.db[col] = OrderedDict({
                        idx: self.lex2idx[col].get(cat_, 0)
                        for idx, cat_ in db_dict[col].items()
                    })
            elif colconfig['type'] in ('text'):
                    self.db[col] = OrderedDict({
                        idx: self.word2idx.get(word, 0)
                        for idx, word in db_dict[col].items()
                    })
            else:
                # Boolean values, numerical values come here
                self.db[col] = OrderedDict(db_dict[col])

    def _initialize_lexicons(self, db_dict, schema_dict):
        '''
            Define columns and metadata about columns
        '''
        def get_basecol(col):
            re_ctxp = r'(_CTX_P)|(_\d)|[\+|\-|\d|]'
            re_repl = r'(_CHILD)|(_PARENT)|(_GRAND_PARENT)'

            base_col = re.sub(re_ctxp, '', col)
            base_col = re.sub(re_repl, '', base_col)

            return base_col

        # Computes a dictionary that maps one column to a base column
        self.columns = self.columns.union(db_dict.keys()).union(set(['INDEX']))
        self.columns_mapper = {col: get_basecol(col) for col in self.columns}

        # Creates descriptors about the data
        dflt_dict = {'category': 'feature', 'type': 'int', 'dims': None}

        # Words are the union of all textual columns
        word_list = []
        lexicons_dict = {}
        for col in list(self.columns):
            lexicon_list = []

            base_col = self.columns_mapper[col]

            config_dict = dict(dflt_dict)
            # Defines metadata
            if base_col in schema_dict:
                config_dict.update(schema_dict[base_col])

            # Lexicon for each column `unk` is not present
            if config_dict['type'] in ('text', 'choice'):  # they are limited
                lexicon_set = set(db_dict[base_col].values())

                # features might be absent ( in case of leading and lagging )
                lexicon_list = sorted(list(lexicon_set))

                if config_dict['type'] in ('text'):
                    word_list += lexicon_list

                if config_dict['category'] in ('feature'):
                    # missing values
                    lexicon_list.insert(0, 'unk')

                elif config_dict['category'] == 'target' and self.filter_br:
                    def filterfn(x):
                        a = re.sub('-C-', '-', x)
                        a = re.sub('C-', '', a)
                        a = re.sub(' ', '', a)
                        return a

                    lexicon_set = set([
                        filterfn(t) for t in lexicon_list
                        if 'A5' not in t or 'AM-MED' not in t
                    ])

                    lexicon_list = sorted(list(lexicon_set))



                config_dict['dims'] = len(lexicon_list)

            self.columns_config[col] = config_dict
            lexicons_dict[col] = lexicon_list

        # filter columns - remove absent columns
        for col in self.columns_config:
            if col not in self.columns:
                del self.columns_config[col]

        # make a single dictonary for text
        word_list = sorted(list(set(word_list)), key= lambda x: x.lower())
        word_list.insert(0, 'unk')

        rng = range(len(word_list))
        self.words = word_list
        self.word2idx = OrderedDict(zip(self.words, rng))
        self.idx2word = OrderedDict(zip(rng, self.words))

        # re-make test columns according to word2idx dict
        for col in self.columns_config:
            col_type = self.columns_config[col]['type']

            if col_type in ('text'):

                self.lex2idx[col] = OrderedDict({
                    lex: self.word2idx[lex]
                    for lex in lexicons_dict[col]
                })

                self.idx2lex[col] = OrderedDict({
                    self.word2idx[lex]: lex
                    for lex in lexicons_dict[col]
                })

            elif col_type in ('choice'):

                self.lex2idx[col] = OrderedDict({
                    lex: i for i, lex in enumerate(lexicons_dict[col])
                })

                self.idx2lex[col] = OrderedDict({
                    i: lex for i, lex in enumerate(lexicons_dict[col])
                })

    def _decode_index(self, idx, columns, encoding):
        d = OrderedDict()

        for col in columns:

            val = self.db[col].get(idx, 0)
            d[col] = self._decode_value(val, col, encoding)

        if len(columns) == 1:
            return d[col]
        else:
            return d

    def _decode_value(self, x, col, encoding):
        col_type = self.columns_config[col]['type']

        if encoding in ('IDX') or col_type in ('int', 'bool'):

            return x

        elif encoding in ('CAT'):

            if col_type in ('choice'):

                return self.idx2lex[col][x]

            elif col_type in ('text'):

                return self.idx2word[x]

            else:

                return x

        elif encoding in ('EMB'):

            if col_type in ('text'):

                w = self.idx2word[x]
                t = self.lex2tok[w]
                i = self.tok2idx[t]

                return self.embeddings[i]

            elif col_type in ('choice'):

                sz = self.column_dimensions(col, encoding)

                return [1 if i == x else 0 for i in range(sz)]

            else:
                return x

        elif encoding in ('HOT'):

            sz = self.column_dimensions(col, encoding)

            if col_type in ('text'):

                w = self.idx2word[x]
                k = self.lex2idx[col].keys()
                j = list(k).index(w)

                return [1 if i == j else 0 for i in range(sz)]

            elif col_type in ('choice'):

                return [1 if i == x else 0 for i in range(sz)]

            else:

                return x

        elif encoding in ('MIX'):

            if col_type in ('text'):

                w = self.idx2word[x]
                t = self.lex2tok[w]
                i = self.tok2idx[t]

                return i

            elif col_type in ('choice'):

                sz = self.column_dimensions(col, encoding)

                return [1 if i == x else 0 for i in range(sz)]

            else:

                return x

        else:
            raise ValueError(
                'No value {x} for {column} using {encoding}'.format(
                    x=x, column=column, encoding=encoding
                )
            )
