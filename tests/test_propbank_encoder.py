'''Tests PropbankEncoder

    Created on July 3, 2018

    @author: Varela
'''
import unittest
from unittest.mock import patch
import os, sys
import json
import yaml
# from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.getcwd()) # import top-level modules
import config
from models.propbank_encoder import PropbankEncoder
# import cps
import utils.corpus as cps

TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
ROOT_DIR = os.getcwd()
FIX_DIR = '{:}fixtures/propbank_encoder/'.format(TESTS_DIR)
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


class PropbankEncoderBaseCase(unittest.TestCase):

    # @patch('cps.fetch_word2vec', return_value=_initialize_word2vec())
    def setUp(self):
        self._initilize()
        cps.fetch_word2vec = unittest.mock.Mock(return_value=self.word2vec)
        cps.get_db_bounds = unittest.mock.Mock(return_value=(1690, 1691))

        # import code; code.interact(local=dict(globals(), **locals()))
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

        lexicon_list = list(self.gs_dict['FORM'].values())
        lexicon_list += self.gs_dict['LEMMA'].values()
        lexicon_list += self.gs_dict['PRED'].values()

        lexicon_list = sorted(list(set(lexicon_list)))
        lexicon_list.insert(0, 'unk')
        self.lexicon_list = lexicon_list


class PropbankTestRecover(PropbankEncoderBaseCase):

    def test_recover(self):
        self.propbank_encoder.persist(FIX_DIR, 'deep_glo50')
        propbank_encoder = PropbankEncoder.recover(PICKLE_PATH)

        mock_dict = self.propbank_encoder.__dict__
        test_dict = propbank_encoder.__dict__

        self.assertEqual(mock_dict, test_dict)


class PropbankTestPersist(PropbankEncoderBaseCase):

    def test_persist(self):
        self.propbank_encoder.persist(FIX_DIR, 'mock_glo50')
        fixture_path = '{:}mock_glo50.pickle'.format(FIX_DIR)
        propbank_encoder = PropbankEncoder.recover(fixture_path)

        mock_dict = self.propbank_encoder.__dict__
        test_dict = propbank_encoder.__dict__

        self.assertEqual(mock_dict, test_dict)
        self.fixtures_list.append(fixture_path)


class PropbankTestWords(PropbankEncoderBaseCase):

    def test_words(self):
        self.assertEqual(self.propbank_encoder.words, self.lexicon_list)


class PropbankTestIdx2Lex(PropbankEncoderBaseCase):

    def test(self):
        for column, meta_dict in self.schema_dict.items():
            if meta_dict['type'] in ('choice', 'text'):
                mock_set = set(self.gs_dict[column].values())
                if meta_dict['category'] in ('feature'):
                    mock_set = mock_set.union(set(['unk']))

                test_set = set(self.propbank_encoder.idx2lex[column].values())

                with self.subTest(msg='Checking {:}'.format(column)):
                    self.assertEqual(mock_set, test_set)


class PropbankTestLex2Idx(PropbankEncoderBaseCase):

    def test(self):
        for column, meta_dict in self.schema_dict.items():
            if meta_dict['type'] in ('choice', 'text'):
                mock_list = list(set(self.gs_dict[column].values()))
                mock_list.sort()

                if meta_dict['category'] in ('feature'):
                    mock_list.insert(0, 'unk')

                if meta_dict['type'] in ('choice'):
                    rng = range(len(mock_list))
                    mock_dict = dict(zip(mock_list, rng))

                else:

                    mock_dict = {
                        word: self.lexicon_list.index(word)
                        for word in mock_list
                    }
                test_dict = dict(self.propbank_encoder.lex2idx[column])
                self.assertEqual(mock_dict, test_dict)


class PropbankTestLex2Tok(PropbankEncoderBaseCase):

    def setUp(self):
        super(PropbankTestLex2Tok, self).setUp()
        self.lex2tok = {lex: lex.lower() for lex in self.lexicon_list}

    def test(self):
        self.assertEqual(self.propbank_encoder.lex2tok, self.lex2tok)


class PropbankTestTokens(PropbankTestLex2Tok):
    def test(self):
        mock_set = set(self.lex2tok.values())
        test_set = set(self.propbank_encoder.tokens)
        self.assertEqual(mock_set, test_set)


class PropbankTestEmbeddings(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestEmbeddings, self).setUp()

        self.embeddings_mock = []
        for word_ in self.lexicon_list:
            tok_ = word_.lower()
            self.embeddings_mock.append(self.word2vec[tok_].tolist())

    def test_embeddings(self):
        embeddings_test = self.propbank_encoder.embeddings
        self.assertEqual(self.embeddings_mock, embeddings_test)

    def test_embeddings_model(self):
        model_test = self.propbank_encoder.embeddings_model
        self.assertEqual(model_test, 'glove')

    def test_embeddings_sz(self):
        embeddings_sz = self.propbank_encoder.embeddings_sz
        self.assertEqual(embeddings_sz, 50)


class PropbankTestColumnIDX(PropbankEncoderBaseCase):

    def setUp(self):
        super(PropbankTestColumnIDX, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function `IDX` encoding'

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
        column_category = self.schema_dict[column_label]['category']

        if column_type in ('choice', 'text'):

            domain_list = list(set(self.gs_dict[column_label].values()))
            domain_list.sort()

            if column_category in ('feature'):
                domain_list.insert(0, 'unk')

            if column_type in ('choice'):
                mock_dict = {
                    idx_: domain_list.index(lex_)
                    for idx_, lex_ in mock_dict.items()
                }
            else:
                mock_dict = {
                    idx_: self.lexicon_list.index(lex_)
                    for idx_, lex_ in mock_dict.items()
                }

        return mock_dict


class PropbankTestColumnCAT(PropbankEncoderBaseCase):

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


class PropbankTestColumnEMB(PropbankEncoderBaseCase):

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
        column_category = self.schema_dict[column_label]['category']

        if column_type in ('choice', 'text'):
            domain_list = sorted(list(set(mock_dict.values())))

            if column_category in ('feature'):
                domain_list.insert(0, 'unk')

            domain_sz = len(domain_list)
            if column_type in ('choice'):
                def make_onehot(x):
                    return [
                        1 if j == domain_list.index(x) else 0
                        for j in range(domain_sz)
                    ]

                mock_dict = {
                    idx_: make_onehot(val_)
                    for idx_, val_ in mock_dict.items()
                }
            else:
                mock_dict = {
                    i: self.word2vec[key.lower()].tolist()
                    for i, key in mock_dict.items()
                }

        return mock_dict


class PropbankTestColumnHOT(PropbankEncoderBaseCase):

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
        column_category = self.schema_dict[column_label]['category']

        if column_type in ('choice', 'text'):
            domain_list = sorted(list(set(mock_dict.values())))
            if column_category in ('feature'):
                domain_list.insert(0, 'unk')

            domain_sz = len(domain_list)

            def make_onehot(x):
                    return [
                        1 if j == domain_list.index(x) else 0
                        for j in range(domain_sz)
                    ]

            if column_type in ('choice', 'text'):
                mock_dict = {
                    idx_: make_onehot(val_)
                    for idx_, val_ in mock_dict.items()
                }

        return mock_dict

class PropbankTestColumnMIX(PropbankEncoderBaseCase):

    def setUp(self):
        super( PropbankTestColumnMIX, self).setUp()
        self.assert_msg = 'Checking label `{:}` for column function `MIX` encoding'

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

        test_dict = self.propbank_encoder.column('train', column_label, 'MIX')
        mock_dict = self._get_mocked_dict(column_label)

        self.assertEqual(test_dict, mock_dict, msg)

    def _get_mocked_dict(self, column_label):
        mock_dict = self.gs_dict[column_label]
        column_type = self.schema_dict[column_label]['type']
        column_category = self.schema_dict[column_label]['category']

        if column_type in ('choice', 'text'):
            domain_list = sorted(list(set(mock_dict.values())))
            if column_category in ('feature'):
                domain_list.insert(0, 'unk')

            domain_sz = len(domain_list)

            def make_onehot(x):
                    return [
                        1 if j == domain_list.index(x) else 0
                        for j in range(domain_sz)
                    ]

            if column_type in ('choice'):
                mock_dict = {
                    idx_: make_onehot(val_)
                    for idx_, val_ in mock_dict.items()
                }
            else:
                mock_dict = {
                    idx_: self.lexicon_list.index(val_)
                    for idx_, val_ in mock_dict.items()
                }

        return mock_dict


class PropbankTestColumnDimensionsCAT(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestColumnDimensionsCAT, self).setUp()
        self.assert_msg = 'propbank_encoder#column_dims(`{:}`, `CAT`) .: should be {:} got {:}'

    # Numerical data
    def test_column_dimensions_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_dimensions_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_dimensions_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_dimensions_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_dimensions_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_dimensions_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_dimensions_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dimensions_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_dimensions_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_dimensions_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_dimensions_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_dimensions_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_dimensions_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        dims_test_ = self.propbank_encoder.column_dimensions(column_label, 'CAT')
        dims_mock_ = 1

        msg = self.assert_msg.format(column_label, dims_mock_, dims_test_)
        self.assertEqual(dims_test_, dims_mock_, msg)


class PropbankTestColumnDimensionsEMB(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestColumnDimensionsEMB, self).setUp()
        self.assert_msg = 'propbank_encoder#column_dims(`{:}`, `EMB`) .: should be {:} got {:}'

    # Numerical data
    def test_column_dimensions_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_dimensions_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_dimensions_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_dimensions_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_dimensions_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_dimensions_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_dimensions_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dimensions_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_dimensions_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_dimensions_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_dimensions_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_dimensions_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_dimensions_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        mock_dict = self.gs_dict[column_label]
        dims_test_ = self.propbank_encoder.column_dimensions(column_label, 'EMB')
        meta_dict = self.schema_dict[column_label]
        column_type = meta_dict['type']
        column_category = meta_dict['category']

        if column_type in ('text',):
            dims_mock_ = 50
        elif column_type in ('choice',):
            dims_mock_ = len(set(mock_dict.values()))  # unknown is a feature
            if column_category in ('feature'):
                dims_mock_ += 1
        else:
            dims_mock_ = 1

        msg = self.assert_msg.format(column_label, dims_mock_, dims_test_)
        self.assertEqual(dims_test_, dims_mock_, msg)


class PropbankTestColumnDimensionsHOT(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestColumnDimensionsHOT, self).setUp()
        self.assert_msg = 'propbank_encoder#column_dims(`{:}`, `HOT`) .: should be {:} got {:}'

    # Numerical data
    def test_column_dimensions_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_dimensions_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_dimensions_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_dimensions_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_dimensions_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_dimensions_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_dimensions_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dimensions_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_dimensions_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_dimensions_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_dimensions_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_dimensions_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_dimensions_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        mock_dict = self.gs_dict[column_label]
        dims_test_ = self.propbank_encoder.column_dimensions(column_label, 'HOT')
        meta_dict = self.schema_dict[column_label]
        column_type = meta_dict['type']
        column_category = meta_dict['category']

        if column_type in ('text', 'choice'):
            dims_mock_ = len(set(mock_dict.values()))
            if column_category in ('feature'):
                dims_mock_ += 1

        else:
            dims_mock_ = 1

        msg = self.assert_msg.format(column_label, dims_mock_, dims_test_)
        self.assertEqual(dims_test_, dims_mock_, msg)


class PropbankTestColumnDimensionsMIX(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestColumnDimensionsMIX, self).setUp()
        self.assert_msg = 'propbank_encoder#column_dims(`{:}`, `MIX`) .: should be {:} got {:}'

    # Numerical data
    def test_column_dimensions_id(self):
        column_label = 'ID'
        self._assert_column_eq(column_label)

    def test_column_dimensions_s(self):
        column_label = 'S'
        self._assert_column_eq(column_label)

    def test_column_dimensions_p(self):
        column_label = 'P'
        self._assert_column_eq(column_label)

    def test_column_dimensions_form(self):
        column_label = 'FORM'
        self._assert_column_eq(column_label)

    def test_column_dimensions_lemma(self):
        column_label = 'LEMMA'
        self._assert_column_eq(column_label)

    def test_column_dimensions_gpos(self):
        column_label = 'GPOS'
        self._assert_column_eq(column_label)

    def test_column_dimensions_morf(self):
        column_label = 'MORF'
        self._assert_column_eq(column_label)

    def test_column_dimensions_dtree(self):
        column_label = 'DTREE'
        self._assert_column_eq(column_label)

    def test_column_dimensions_func(self):
        column_label = 'FUNC'
        self._assert_column_eq(column_label)

    def test_column_dimensions_pred(self):
        column_label = 'PRED'
        self._assert_column_eq(column_label)

    def test_column_dimensions_arg(self):
        column_label = 'ARG'
        self._assert_column_eq(column_label)

    def test_column_dimensions_iob(self):
        column_label = 'IOB'
        self._assert_column_eq(column_label)

    def test_column_dimensions_t(self):
        column_label = 'T'
        self._assert_column_eq(column_label)

    def _assert_column_eq(self, column_label):
        mock_dict = self.gs_dict[column_label]
        dims_test_ = self.propbank_encoder.column_dimensions(column_label, 'MIX')
        meta_dict = self.schema_dict[column_label]
        column_type = meta_dict['type']
        column_category = meta_dict['category']

        if column_type in ('choice'):
            dims_mock_ = len(set(mock_dict.values()))
            if column_category in ('feature'):
                dims_mock_ += 1

        else:
            dims_mock_ = 1

        msg = self.assert_msg.format(column_label, dims_mock_, dims_test_)
        self.assertEqual(dims_test_, dims_mock_, msg)

class PropbankTestIteratorIDX(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestIteratorIDX, self).setUp()
        _params = ('train', ['ID', 'FORM', 'ARG'], 'IDX')
        self.iterator = self.propbank_encoder.iterator(*_params)
        self.base_message = 'got {:} should be {:}'

    def test(self):
        for index, column_dict in self.iterator:
            for column_label, column_value in column_dict.items():
                msg_ = 'index: {:} label:`{:}`'.format(index, column_label)

                with self.subTest(message=msg_):
                    mock_value = self.gs_dict[column_label][index]
                    if column_label in ('FORM', 'ARG'):
                        domain_list = list(set(self.gs_dict[column_label].values()))
                        if column_label in ('FORM'):
                            mock_value = self.lexicon_list.index(mock_value)
                        else:
                            mock_value = sorted(domain_list).index(mock_value)

                    msg_ = self.base_message.format(column_value, mock_value)
                    self.assertEqual(column_value, mock_value, msg_)


class PropbankTestIteratorCAT(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestIteratorCAT, self).setUp()
        _params = ('train', ['ID', 'FORM', 'ARG'], 'CAT')
        self.iterator = self.propbank_encoder.iterator(*_params)
        self.base_message = 'got `{:}` should be `{:}`'

    def test(self):
        for index, column_dict in self.iterator:
            for column_label, column_value in column_dict.items():
                msg_ = 'index: {:}     label:`{:}`'.format(index, column_label)
                with self.subTest(message=msg_):
                    mock_value = self.gs_dict[column_label][index]

                    msg_ = self.base_message.format(column_value, mock_value)
                    self.assertEqual(column_value, mock_value, msg_)


class PropbankTestIteratorEMB(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestIteratorEMB, self).setUp()
        _params = ('train', ['ID', 'FORM', 'ARG'], 'EMB')
        self.iterator = self.propbank_encoder.iterator(*_params)
        self.base_message = 'got {:} should be {:}'

    def test(self):
        for index, column_dict in self.iterator:
            for column_label, column_value in column_dict.items():
                msg_ = 'index: {:}  label:`{:}`'.format(index, column_label)
                with self.subTest(message=msg_):
                    mock_value = self.gs_dict[column_label][index]
                    if column_label in ('ARG'):
                        domain_list = sorted(list(set(self.gs_dict[column_label].values())))
                        i = domain_list.index(mock_value)
                        sz = len(domain_list)
                        mock_value = [1 if i == j else 0 for j in range(sz)]
                    elif column_label in ('FORM',):
                        mock_value = self.word2vec[mock_value.lower()]
                        mock_value = mock_value.tolist()

                    msg_ = self.base_message.format(column_value, mock_value)
                    self.assertEqual(column_value, mock_value, msg_)


class PropbankTestIteratorMIX(PropbankEncoderBaseCase):
    def setUp(self):
        super(PropbankTestIteratorMIX, self).setUp()
        _params = ('train', ['ID', 'FORM', 'ARG'], 'MIX')
        self.iterator = self.propbank_encoder.iterator(*_params)
        self.base_message = 'got {:} should be {:}'

    def test(self):
        for index, column_dict in self.iterator:
            for column_label, column_value in column_dict.items():
                msg_ = 'index: {:}  label:`{:}`'.format(index, column_label)
                with self.subTest(message=msg_):
                    mock_value = self.gs_dict[column_label][index]
                    if column_label in ('ARG'):
                        domain_list = sorted(list(set(self.gs_dict[column_label].values())))
                        i = domain_list.index(mock_value)
                        sz = len(domain_list)
                        mock_value = [1 if i == j else 0 for j in range(sz)]
                    elif column_label in ('FORM',):
                        mock_value = self.lexicon_list.index(mock_value)

                    msg_ = self.base_message.format(column_value, mock_value)
                    self.assertEqual(column_value, mock_value, msg_)


if __name__ == '__main__':
    unittest.main()