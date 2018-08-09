'''Tests EvaluatorConll

    Created on July 8, 2018

    @author: Varela
'''
import unittest
import os, sys
sys.path.insert(0, os.getcwd())

from models.evaluator_conll import EvaluatorConll
from models.propbank_encoder import PropbankEncoder
# from tests.test_propbank_encoder import PropbankEncoderBaseCase


TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
FIX_DIR = '{:}fixtures/'.format(TESTS_DIR)
FIX_EVALUATOR_DIR = '{:}evaluator_conll/'.format(FIX_DIR)
FIX_PROPBANK_DIR = '{:}propbank_encoder/'.format(FIX_DIR)
FIX_PROPBANK_PATH = '{:}deep_glo50.pickle'.format(FIX_PROPBANK_DIR)


TEST_PATH = '{:}lr1.00e-03_hs32x32_ctx-p1/'.format(FIX_EVALUATOR_DIR)


class EvaluatorConllBaseCase(unittest.TestCase):
    def setUp(self):
        self.pe = PropbankEncoder.recover(FIX_PROPBANK_PATH)

        self.evaluator = EvaluatorConll(
            self.pe.db,
            self.pe.idx2lex,
            target_dir=FIX_DIR)

        params_ = ('train', 'ARG', 'CAT')
        self.gs_props = self.pe.column(*params_)

    def tearDown(self):
        try:
            if os.path.isdir(TEST_PATH):
                os.remove(TEST_PATH)
        except Exception:
            pass

class EvaluatorConllF1(EvaluatorConllBaseCase):

    def test_100(self):
        self.evaluator.evaluate('test', self.gs_props, {})
        self.assertEqual(self.evaluator.f1, 100.0)

    def test_50(self):
        props_50 = {}
        for i, key in enumerate(self.gs_props):
            if i == 2:
                props_50[key] = '*)'
            elif i > 2 and i < 6:
                props_50[key] = '*'
            else:
                props_50[key] = self.gs_props[key]
        self.evaluator.evaluate('test', props_50, {})
        self.assertEqual(self.evaluator.f1, 50.0)

    def test_00(self):
        props_00 = {key: '*' for key in self.gs_props}
        self.evaluator.evaluate('test', props_00, {})
        self.assertEqual(self.evaluator.f1, 0.0)
