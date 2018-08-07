'''Tests conll read file

    Created on July 3, 2018

    @author: Varela
'''
import unittest
from unittest.mocks import patch
import os, sys
import json
import yaml


import numpy as np

sys.path.insert(0, os.getcwd()) # import top-level modules
import config
from models.propbank_encoder import PropbankEncoder

TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
ROOT_DIR = os.getcwd()
MOCK_DIR = '{:}mocks/'.format(TESTS_DIR)


def _get_word2vec():
    path_ = '{:}word2vec.json'.format(MOCK_DIR)
    with open(path_, mode='r') as f:
        json_dict = json.load(f)

    return {w: np.array(v) for w, v in json_dict.items()}


def _get_gs():
    path_ = '{:}gs.json'.format(MOCK_DIR)
    with open(path_, mode='r') as f:
        gs_json = json.load(f)

    return {keycol_: {
                        int(key_): val_ 
                        for key_, val_ in innerdict_.items()
                     }
                for keycol_, innerdict_ in gs_json.items()
            }

def _get_schema():
    schema_path = '{:}/{:}gs.yaml'.format(root_dir, config.SCHEMA_DIR)
    with open(schema_path, mode='r') as f:
            schema_dict = yaml.load(f)

    return schema_dict


class PropbankTest(unittest.TestCase):

    @patch('utl.fetch_word2vec', return_value=_get_word2vec())
    def setUp(self):
        self.gs_dict = _get_gs()
        self.schema_dict = _get_schema()

        self.propbank_encoder = \
            PropbankEncoder(self.gs_dict, self.schema_dict, language_model='glove_s50')

    def test_value(self):
        print(self.propbank_encoder.db)
        self.assertEqual(True, True)

        

if __name__ == '__main__':
    unittest.main()