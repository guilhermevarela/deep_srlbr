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


def test_maker_eq(test, mock, msg):
    '''Returns a maker

    [description]

    Arguments:
        test {<comparable>} -- value to test for equality (implements __eq__)
        mock {<comparable>} -- value to compare to (implements __eq__)
        msg {str} -- message

    Returns:
        [type] -- [description]
    '''
    def test(self):
        self.assertEqual(test, mock, msg)
    return test


def test_binder_eq(obj, test_list, mock_list, name_list):
    '''Binds a test to an object

    This method attaches a test to the class

    Arguments:
        obj {[type]} -- [description]
        test_list {list} -- [description]
        mock_list {[type]} -- [description]
        name_list {[type]} -- [description]
    '''
    zip_list = zip(name_list, test_list, mock_list)
    for name_, test_, mock_ in zip_list:
        fnc = test_maker_eq(test_, mock_, name_)
        setattr(obj, 'test_{:}'.format(name_), fnc)


class PropbankBaseCase(unittest.TestCase):

    # @patch('models.utils.fetch_word2vec', return_value=_initialize_word2vec())
    def setUp(self):
        self._initilize()
        models.utils.fetch_word2vec = unittest.mock.Mock(return_value=self.word2vec)
        models.utils.get_db_bounds = unittest.mock.Mock(return_value=(1690, 1691))
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

    def _initilize(self):
        self._initialize_gs()
        self._initialize_schema()
        self._initialize_word2vec()

    def _initialize_word2vec(self):
        path_ = '{:}word2vec.json'.format(FIX_DIR)
        with open(path_, mode='r') as f:
            json_dict = json.load(f)

        self.word2vec = {w: np.array(v) for w, v in json_dict.items()}

    def _initialize_gs(self):
        path_ = '{:}gs.json'.format(FIX_DIR)
        with open(path_, mode='r') as f:
            gs_json = json.load(f)

        self.gs_dict = {
            keycol_: {int(key_): val_ for key_, val_ in innerdict_.items()}
            for keycol_, innerdict_ in gs_json.items()}

    def _initialize_schema(self):
        schema_path = '{:}schema.yml'.format(FIX_DIR)
        with open(schema_path, mode='r') as f:
                schema_dict = yaml.load(f)

        self.schema_dict = schema_dict


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

    def test(self):
        for column, meta_dict in self.schema_dict.items():
            if meta_dict['type'] in ('choice', 'text'):
                mock_ = meta_dict['domain']
                test_values_ = self.propbank_encoder.idx2lex[column].values()

                with self.subTest(msg='Checking {:}'.format(column)):
                    self.assertEqual(list(test_values_), list(mock_))


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


class PropbankTestEmbeddings(PropbankBaseCase):
    def setUp(self):
        super(PropbankTestEmbeddings, self).setUp()

        tokens = list(self.word2vec.keys())
        self.embeddings_test = [self.word2vec['unk'].tolist()]
        for tok_ in sorted(tokens):
            if tok_ != 'unk':
                self.embeddings_test.append(self.word2vec[tok_].tolist())

    def test_embeddings(self):
        self.assertEqual(self.propbank_encoder.embeddings, self.embeddings_test)

    def test_embeddings_model(self):
        self.assertEqual(self.propbank_encoder.embeddings_model, 'glove')

    def test_embeddings_sz(self):
        self.assertEqual(self.propbank_encoder.embeddings_sz, 50)


class PropbankTestColumnCAT(PropbankBaseCase):

    def setUp(self):
        super(PropbankTestColumnCAT, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function index encoding'

    def test_column_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)        

    def _assert_column_eq(self, column_label):
        msg = self.assert_msg.format(column_label)

        test_dict = self.propbank_encoder.column('train', column_label, 'CAT')
        mock_dict = self.gs_dict[column_label]

        self.assertEqual(test_dict, mock_dict, msg)


class PropbankTestColumnEMB(PropbankBaseCase):

    def setUp(self):
        super(PropbankTestColumnEMB, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function `EMB` encoding'

    def test_column_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        msg = self.assert_msg.format(column_label)

        test_dict = self.propbank_encoder.column('train', column_label, 'EMB')
        mock_dict = self._get_mocked_dict(column_label)

        self.assertEqual(test_dict, mock_dict, msg)

    def _get_mocked_dict(self, column_label):
        mock_dict = self.gs_dict[column_label]
        column_type = self.schema_dict[column_label]['type']

        if column_type in ('choice'):
            domain_ = self.schema_dict[column_label]['domain']
            domain_list = list(domain_)
            domain_sz = len(domain_)
            lex2list_dict = {
                key: [1 if i == j else 0 for j in range(domain_sz)]
                for i, key in enumerate(domain_list)                
            }
            mock_dict = {key_: lex2list_dict[lex_]
                for key_, lex_ in mock_dict.items()
            }
        elif column_type in ('text',):
            domain_ = self.schema_dict[column_label]['domain']
            domain_list = list(domain_)
            domain_sz = len(domain_)
            mock_dict = {
                i: self.word2vec[key.lower()].tolist()
                for i, key in mock_dict.items()
            }

        return mock_dict

class PropbankTestColumnIDX(PropbankBaseCase):

    def setUp(self):
        super(PropbankTestColumnIDX, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function index encoding'

    def test_column_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        msg = self.assert_msg.format(column_label)

        test_dict = self.propbank_encoder.column('train', column_label, 'IDX')
        mock_dict = self._get_mocked_dict(column_label)

        self.assertEqual(test_dict, mock_dict, msg)

    def _get_mocked_dict(self, column_label):
        mock_dict = self.gs_dict[column_label]
        column_type = self.schema_dict[column_label]['type']

        if column_type in ('choice', 'text'):
            domain_ = self.schema_dict[column_label]['domain']
            domain_list = list(domain_)
            mock_dict = {key_: domain_list.index(lex_)
                for key_, lex_ in mock_dict.items()
            }
        
        return mock_dict


class PropbankTestColumnHOT(PropbankBaseCase):

    def setUp(self):
        super(PropbankTestColumnHOT, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function index encoding'

    def test_column_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        msg = self.assert_msg.format(column_label)

        test_dict = self.propbank_encoder.column('train', column_label, 'HOT')
        mock_dict = self._get_mocked_dict(column_label)

        self.assertEqual(test_dict, mock_dict, msg)

    def _get_mocked_dict(self, column_label):
        mock_dict = self.gs_dict[column_label]
        column_type = self.schema_dict[column_label]['type']

        if column_type in ('choice', 'text'):
            domain_ = self.schema_dict[column_label]['domain']
            domain_list = list(domain_)
            domain_sz = len(domain_)
            mock_dict = {
                key_: [1 if i == domain_list.index(lex_) else 0 for i in range(domain_sz)]
                for key_, lex_ in mock_dict.items()}

        return mock_dict



if __name__ == '__main__':
    unittest.main()