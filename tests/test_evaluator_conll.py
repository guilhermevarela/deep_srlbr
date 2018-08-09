'''Tests EvaluatorConll

    Created on July 8, 2018

    @author: Varela
'''
import unittest
import os, sys
sys.path.insert(0, os.getcwd())

from models.evaluator_conll import EvaluatorConll
from tests.test_propbank_encoder import PropbankEncoderBaseCase


TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
FIX_DIR = '{:}fixtures/evaluator_conll/'.format(TESTS_DIR)
TEST_PATH = '{:}lr1.00e-03_hs32x32_ctx-p1/test.conll'.format(FIX_DIR)

class EvaluatorConllBaseCase(PropbankEncoderBaseCase):
    def setUp(self):
        super(EvaluatorConllBaseCase, self).setUp()
        self.evaluator = EvaluatorConll(
            self.propbank_encoder.db,
            self.propbank_encoder.idx2lex,
            target_dir=FIX_DIR)

        self.evaluator.evaluate('test', self.gs_dict['ARG'], {})

    def test_f1(self):
        self.assertEqual(self.evaluator.f1, 100)

    def test_conll(self):
