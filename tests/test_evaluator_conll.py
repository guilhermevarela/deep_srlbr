'''Tests EvaluatorConll

    Created on July 8, 2018

    @author: Varela
'''
import unittest
import os, sys
sys.path.insert(0, os.getcwd())

import numpy as np

from models.evaluator_conll import EvaluatorConll
from models.propbank_encoder import PropbankEncoder


TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
FIX_DIR = '{:}fixtures/'.format(TESTS_DIR)
FIX_EVALUATOR_DIR = '{:}evaluator_conll/'.format(FIX_DIR)
FIX_EVALUATOR_MOCK_DIR = '{:}mocks/'.format(FIX_EVALUATOR_DIR)
FIX_PROPBANK_DIR = '{:}propbank_encoder/'.format(FIX_DIR)
FIX_PROPBANK_PATH = '{:}deep_glo50.pickle'.format(FIX_PROPBANK_DIR)

FIX_TMP_DIR = '{:}lr1.00e-03_hs32x32_ctx-p1/'.format(FIX_EVALUATOR_DIR)


def dict2npy(target_dict):
    '''Converts a prop_dict into numpy arrays

    Helper function to convert, dict_keys and dict_values to numpy


    Arguments:
        target_dict {dict} -- A proposition target dictionary

    Returns:
        [type] -- [description]
    '''
    npy_index = np.array(list(target_dict.keys()))
    npy_index = np.expand_dims(npy_index, axis=1) # column array
    npy_predictions = np.array(
        [ary_ for _, ary_ in target_dict.items()]
    )
    # Add batch_size of one
    npy_predictions = np.expand_dims(npy_predictions, axis=1) # column array
    return npy_index, npy_predictions


class EvaluatorConllBaseCase(unittest.TestCase):
    def setUp(self):
        self.pe = PropbankEncoder.recover(FIX_PROPBANK_PATH)

        self.evaluator = EvaluatorConll(
            self.pe.db,
            self.pe.idx2lex,
            target_dir=FIX_EVALUATOR_DIR)

        params_ = ('train', 'ARG', 'CAT')
        self.gs_props = self.pe.column(*params_)

    def tearDown(self):
        try:
            if os.path.isdir(FIX_TMP_DIR):
                os.remove(FIX_TMP_DIR)
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


class EvaluatorConllFromConllFile(EvaluatorConllBaseCase):
        def setUp(self):
            super(EvaluatorConllFromConllFile, self).setUp()
            self.target_path = '{:}mock.conll'.format(FIX_EVALUATOR_MOCK_DIR)

        def test(self):
            self.evaluator.evaluate_fromconllfile(self.target_path)
            self.assertEqual(self.evaluator.f1, 100.0)


class EvaluatorConllTensor(EvaluatorConllBaseCase):

        def setUp(self):
            super(EvaluatorConllTensor, self).setUp()
            self.prefix = 'test'
            self.index = np.array(list(self.gs_props.keys()))
            self.batch_size = 1
            self.hparams = {}
            # evaluate_tensor(self, prefix, index_tensor, predictions_tensor, len_tensor, target_column, hparams):

        def test_arguments(self):
            target_dict = self.pe.column('train', 'ARG', 'IDX')
            depth = len(target_dict)
            width = self.pe.column_dimensions('ARG', 'HOT')  # target dimensions

            npy_index, npy_predictions = dict2npy(target_dict)

            self.evaluator.evaluate_tensor(
                self.prefix, npy_index, npy_predictions, [width] * depth, 'ARG', {})

            self.assertEqual(self.evaluator.f1, 100.0)

        def test_iob(self):
            target_dict = self.pe.column('train', 'IOB', 'IDX')
            depth = len(target_dict)
            width = self.pe.column_dimensions('IOB', 'HOT')  # target dimensions

            npy_index, npy_predictions = dict2npy(target_dict)

            self.evaluator.evaluate_tensor(
                self.prefix, npy_index, npy_predictions, [width] * depth, 'IOB', {})

            self.assertEqual(self.evaluator.f1, 100.0)

        def test_iob(self):
            target_dict = self.pe.column('train', 'IOB', 'IDX')
            depth = len(target_dict)
            width = self.pe.column_dimensions('IOB', 'HOT')  # target dimensions

            npy_index, npy_predictions = dict2npy(target_dict)

            self.evaluator.evaluate_tensor(
                self.prefix, npy_index, npy_predictions, [width] * depth, 'IOB', {})

            self.assertEqual(self.evaluator.f1, 100.0)            
