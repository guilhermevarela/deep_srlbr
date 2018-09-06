'''Test utils functions

    Created on Sep 7, 2018

'''
import unittest

import pandas as pd

from utils.chunk import chunk_dict_maker
EX_PATH = 'tests/fixtures/chunks/ex.txt'


class ChunkTest(unittest.TestCase):
    def setUp(self):
        params = {'sep':',', 'index_col':0 ,'header': 0, 'encoding': 'utf-8'}
        db_dict = pd.read_csv(EX_PATH, **params).to_dict()
        keys = list(db_dict['FORM'].keys())
        id_dict = dict(zip(keys, list(keys)))
        prop_dict = dict(zip(keys, [1] * len(keys)))
        self.form_dict = db_dict['FORM']
        self.ctree_dict = db_dict['CTREE']
        self.ckiob_dict = chunk_dict_maker(id_dict, prop_dict, self.ctree_dict)[1]

    def test_A_should_be_B_NP(self):
        msg = '{:} should be `B-NP` got {:}'.format(*self._get_test_ex(1))
        self.assertEqual(self.ckiob_dict[1], 'B-NP', msg)

    def test_falta_should_be_I_NP(self):
        msg = '{:} should be `I-NP` got {:}'.format(*self._get_test_ex(2))
        self.assertEqual(self.ckiob_dict[2], 'I-NP', msg)

    def test_de_should_be_B_PP_1(self):
        msg = '{:} should be `B-PP` got {:}'.format(*self._get_test_ex(3))
        self.assertEqual(self.ckiob_dict[3], 'B-PP', msg)

    def test_um_should_be_I_PP(self):
        msg = '{:} should be `I-PP` got {:}'.format(*self._get_test_ex(4))
        self.assertEqual(self.ckiob_dict[4], 'I-PP', msg)

    def test_programa_should_be_I_PP(self):
        msg = '{:} should be `I-PP` got {:}'.format(*self._get_test_ex(5))
        self.assertEqual(self.ckiob_dict[5], 'I-PP', msg)

    def test_poderia_should_be_B_VP(self):
        msg = '{:} should be `B-VP` got {:}'.format(*self._get_test_ex(6))
        self.assertEqual(self.ckiob_dict[6], 'B-VP', msg)

    def test_dar_should_be_I_VP(self):
        msg = '{:} should be `I-VP` got {:}'.format(*self._get_test_ex(7))
        self.assertEqual(self.ckiob_dict[7], 'I-VP', msg)

    def test_a_should_be_B_NP(self):
        msg = '{:} should be `B-NP` got {:}'.format(*self._get_test_ex(8))
        self.assertEqual(self.ckiob_dict[8], 'B-NP', msg)

    def test_impressao_should_be_I_NP(self):
        msg = '{:} should be `I-NP` got {:}'.format(*self._get_test_ex(9))
        self.assertEqual(self.ckiob_dict[9], 'I-NP', msg)

    def test_de_should_be_B_PP_2(self):
        msg = '{:} should be `B-PP` got {:}'.format(*self._get_test_ex(10))
        self.assertEqual(self.ckiob_dict[10], 'B-PP', msg)

    def test_nao_should_be_B_ADVP(self):
        msg = '{:} should be `B-ADVP` got {:}'.format(*self._get_test_ex(11))
        self.assertEqual(self.ckiob_dict[11], 'B-ADVP', msg)

    def test_ter_should_be_B_VP(self):
        msg = '{:} should be `B-VP` got {:}'.format(*self._get_test_ex(12))
        self.assertEqual(self.ckiob_dict[12], 'B-VP', msg)

    def test_metas_should_be_B_NP(self):
        msg = '{:} should be `B-NP` got {:}'.format(*self._get_test_ex(13))
        self.assertEqual(self.ckiob_dict[13], 'B-NP', msg)

    def test_dot_should_be_O(self):
        msg = '{:} should be `O` got {:}'.format(*self._get_test_ex(14))
        self.assertEqual(self.ckiob_dict[14], 'O', msg)

    def _get_test_ex(self, k):
        return (self.form_dict[k], self.ckiob_dict[k])