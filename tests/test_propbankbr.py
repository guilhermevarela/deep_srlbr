'''Tests conll read file

    Created on July 3, 2018

    @author: Varela
'''
import unittest
from unittest.mock import patch
import os, sys
import json
import yaml


import numpy as np

sys.path.insert(0, os.getcwd()) # import top-level modules
import config
from models.propbank_encoder import PropbankEncoder
import models.utils

TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
ROOT_DIR = os.getcwd()
FIX_DIR = '{:}fixtures/'.format(TESTS_DIR)
PICKLE_PATH = '{:}deep_glo50.pickle'.format(FIX_DIR)


def _get_word2vec():
    path_ = '{:}word2vec.json'.format(FIX_DIR)
    with open(path_, mode='r') as f:
        json_dict = json.load(f)

    return {w: np.array(v) for w, v in json_dict.items()}


def _get_gs():
    path_ = '{:}gs.json'.format(FIX_DIR)
    with open(path_, mode='r') as f:
        gs_json = json.load(f)

    return {
        keycol_: {int(key_): val_ for key_, val_ in innerdict_.items()}
        for keycol_, innerdict_ in gs_json.items()}


def _get_schema():
    schema_path = '{:}schema.yml'.format(FIX_DIR)
    with open(schema_path, mode='r') as f:
            schema_dict = yaml.load(f)

    return schema_dict


class PropbankBaseCase(unittest.TestCase):

    # @patch('models.utils.fetch_word2vec', return_value=_get_word2vec())
    def setUp(self):
        self.gs_dict = _get_gs()
        self.schema_dict = _get_schema()
        self.word2vec = _get_word2vec()
        models.utils.fetch_word2vec = unittest.mock.Mock(return_value=self.word2vec)
        self.propbank_encoder = \
            PropbankEncoder(self.gs_dict, self.schema_dict, language_model='glove_s50', verbose=False)

        self.fixtures_list = []

    def tearDown(self):
        if self.fixtures_list:
            for fix_path in self.fixtures_list:
                try:
                    os.remove(fix_path)
                except Exception:
                    pass

class PropbankTestRecover(PropbankBaseCase):

    def test_recover(self):
        self.propbank_encoder.persist(FIX_DIR, 'deep_glo50')
        propbank_encoder = PropbankEncoder.recover(PICKLE_PATH)

        self.assertEqual(self.propbank_encoder.__dict__, propbank_encoder.__dict__)


class PropbankTestPersist(PropbankBaseCase):

    def test_persist(self):
        self.propbank_encoder.persist(FIX_DIR, 'mock_glo50')
        fixture_path = '{:}mock_glo50.pickle'.format(FIX_DIR)
        propbank_encoder = PropbankEncoder.recover(fixture_path)

        self.assertEqual(self.propbank_encoder.__dict__, propbank_encoder.__dict__)
        self.fixtures_list.append(fixture_path)


class PropbankTestWords(PropbankBaseCase):

    def test_words(self):
        words_test = self.schema_dict['LEMMA']['domain']
        words_test += self.schema_dict['FORM']['domain']
        words_test_set = set(words_test)
        self.assertEqual(self.propbank_encoder.words, words_test_set)


class PropbankTestIdx2Lex(PropbankBaseCase):

    def test_idx2lex(self):
        for column, meta_dict in self.schema_dict.items():
            if meta_dict['type'] == 'choice':
                with self.subTest(msg='Checking {:}'.format(column)):
                    self._helper_choice(column)

            elif meta_dict['type'] == 'text':
                with self.subTest(msg='Checking {:}'.format(column)):
                    self._helper_text(column)


    def _helper_choice(self, choice_column):
        domain_ = self.schema_dict[choice_column]['domain']
        test_dict = {i:d for i, d in enumerate(domain_)}

        self.assertEqual(test_dict,
                         self.propbank_encoder.idx2lex[choice_column])

    def _helper_text(self, text_column):
        domain_ = self.schema_dict[text_column]['domain']
        test_dict = {i:d for i, d in enumerate(domain_)}

        self.assertEqual(test_dict,
                         self.propbank_encoder.idx2lex[text_column])

class PropbankTestLex2Idx(PropbankBaseCase):

    def test_lex2idx(self):
        for column, meta_dict in self.schema_dict.items():
            if meta_dict['type'] in ('choice', 'text'):
                keys_ = self.propbank_encoder.idx2lex[column].keys()
                values_ = self.propbank_encoder.idx2lex[column].values()
                test_dict = dict(zip(values_, keys_))
                self.assertEqual(test_dict,
                                 self.propbank_encoder.lex2idx[column])

class PropbankTestLex2Tok(PropbankBaseCase):

    def setUp(self):
        super(PropbankTestLex2Tok, self).setUp()
        lexicon_list = self.schema_dict['FORM']['domain'] 
        lexicon_list += self.schema_dict['LEMMA']['domain']
        sorted(lexicon_list)
        self.lex2tok = {lex: lex.lower() for lex in list(set(lexicon_list))}

    def test(self):
        self.assertEqual(self.propbank_encoder.lex2tok, self.lex2tok)


class PropbankTestTokens(PropbankTestLex2Tok):
    def test(self):
        self.assertEqual(self.propbank_encoder.tokens, set(self.lex2tok.values()))

# class PropbankTestEmbeddings(PropbankBaseCase):
#     def setUp(self):
#         super(PropbankTestEmbeddings, self).setUp()

#         zip_list = [(key_, val_) for key_, val_ in self.word2vec]
#         sorted(zip_list, lambda x: x[0])

#         self.embeddings_test = [self.word2vec['unk'].tolist()]
#         for w_, v_ in zip_list:
#             if w_ != 'unk':
#                 self.embeddings_test.append(v_.tolist())

#     def test_embeddings(self):
#         print(self.propbank_encoder.embeddings)
#         print(self.embeddings_test)
#         self.assertEqual(self.propbank_encoder.embeddings,  self.embeddings_test)

#     def test_embeddings_model(self):
#         self.assertEqual(self.propbank_encoder.embeddings_model,  'glove')
    
    def test_embeddings_sz(self):
        self.assertEqual(self.propbank_encoder.embeddings_sz,  50)
