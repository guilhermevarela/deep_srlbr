'''
    Created on Apr 20, 2018
        @author: Varela

    PropbankEncoder wrapps a set of columns indexing them
'''
import re
import yaml
import numpy as np
import sys
sys.path.append('..')
import config
from collections import OrderedDict, defaultdict
from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess



SCHEMA_PATH = '../{:}gs.yaml'.format(config.SCHEMA_DIR)
# EMBEDDINGS_PATH = '../{:'.format(config.LANGUAGE_MODEL_DIR)


class PropbankEncoder(object):
    '''
        PropbankEncoder mantains an indexed representation of the database
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
    data = defaultdict(OrderedDict)
    encodings = ('CAT', 'IDX', 'HOT', 'EMB')
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

    def define(self, dbpt_d, language_model='glove_s50.txt', verbose=True):

        # RUNS EMBEDDINGS
        self._process_embeddings(language_model, verbose)


    def _process_embeddings(self, language_model, verbose):
        # computes embeddings
        word2vec = fetch_word2vec(language_model=language_model, verbose=verbose)
        self.embeddings_model, self.embeddings_sz = language_model.split('_s')

        self.lex2tok = preprocess(list(self.lexicon), word2vec)
        if verbose:
            words = self.lex2tok.keys()
            tokens = set(self.lex2tok.values())
            print('# UNIQUE TOKENIZED {:}, # EMBEDDED TOKENS {:}'.format(len(words), len(tokens)))

        embeddings_shape = (len(tokens), self.embeddings_sz)
        self.tok2idx = {'unk': 0}
        self.idx2tok = {0: 'unk'}
        self.embeddings = np.zeros(embeddings_shape, dtype=np.float32)
        self.embeddings[0] = word2vec['unk']

        i = 1
        for tok in list(tokens):
            if not tok in word2vec:
                self.tok2idx[tok] = i
                self.idx2tok[i] = tok
                self.embeddings[i] = word2vec[tok]
                i += 1


    def _process_db(self, dbpt_d):
        # Computes a dictionary that maps one column to a base column
        self.columns = self.columns.union(self.dbpt_d.keys())
        self.columns_mapper = {col: re.sub(r'[\+|\-|\d|]|(_CTX_P)', '', col)
                               for col in self.columns}

        for col in list(self.columns):
            base_col = columns_mapper[col]
            if col in ('INDEX') or not base_col in self.schema_d:
                #Boolean values, numerical values come here
                self.data[col] = OrderedDict(self.dbpt_d[col])
            else:
                if self.schema_d[base_col] in ('choice'):
                    if not base_col in self.onehot:
                        if base_col in self.schema_d
                            d = self.schema_d[base_col]['domain']
                        else:
                            d = set(self.dbpt_d[col].values())

                    self.data[col] = OrderedDict({
                        idx: self.onehot[base_col][idx] for idx, category in self.dbpt_d[col]
                    })

                elif self.schema_d[base_col] in ('str'):
                    self.data[col] = OrderedDict({
                        idx: self.tok2idx[self.lex2tok[word]] for idx, word in self.dbpt_d[col]
                    })
                else:
                    #Boolean values, numerical values come here










if __name__ == '__main__':    
    propbank_encoder = PropbankEncoder()
    import code; code.interact(local=dict(globals(), **locals()))
    propbank_encoder.schema_d

