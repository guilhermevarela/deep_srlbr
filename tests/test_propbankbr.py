'''Tests datasets.scripts.propbank module

    Created on Aug 23, 2018

    @author: Varela
'''
import sys, os
import unittest
sys.path.insert(0, os.getcwd()) # import top-level modules

from datasets.scripts import propbankbr as br
from models.evaluator_conll import _props_file2zip_list

TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
FIX_DIR = '{:}fixtures/'.format(TESTS_DIR)
FIX_EVALUATOR_DIR = '{:}evaluator_conll/'.format(FIX_DIR)
FIX_EVALUATOR_MOCK_DIR = '{:}mocks/'.format(FIX_EVALUATOR_DIR)
FIX_PROPBANK_DIR = '{:}propbank_encoder/'.format(FIX_DIR)
FIX_PROPBANK_PATH = '{:}deep_glo50.pickle'.format(FIX_PROPBANK_DIR)

FIX_TMP_DIR = '{:}lr1.00e-03_hs32x32_ctx-p1/'.format(FIX_EVALUATOR_DIR)


class PropBankBrTestCase(unittest.TestCase):
    def setUp(self):
        gold_fix_path = '{:}/mock-gold.props'.format(FIX_EVALUATOR_MOCK_DIR)
        gold_list = _props_file2zip_list(gold_fix_path)

        self.se_gold = br.propbankbr_arg2se(gold_list)

    def test_open_a0(self):
        self.assertEqual(self.se_gold[0][1], '(A0*')

    def test_close_a0(self):
        self.assertEqual(self.se_gold[5][1], '*A0)')

    def test_verb(self):
        self.assertEqual(self.se_gold[6][1], '(V*V)')

    def test_enclosed_a1(self):
        self.assertEqual(self.se_gold[7][1], '(A1*A1)')

# class PropBankBrArg2SE(PropBankBrTestCase):
#     def setUp(self):
#         # super(PropBankBrTestCase, self).setUp()
#         self.se_gold = br.propbankbr_arg2se(self.prop_list,
#                                             self.gold_list)

#     def test_open_a0(self):
#         self.assertEqual(self.se_gold[0][0], '(A0*')

#     def test_close_a0(self):
#         self.assertEqual(self.se_gold[0][5], '*A0)')

#     def test_verb(self):
#         self.assertEqual(self.se_gold[0][6], '(V*V)')

#     def test_enclosed_a1(self):
#         self.assertEqual(self.se_gold[0][7], '(A1*A1)')
