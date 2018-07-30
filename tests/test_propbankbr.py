'''Tests conll read file

    Created on July 3, 2018

    @author: Varela
'''
import unittest
import os
import json
import yaml

import numpy as np
import config
from models.propbank_encoder import PropbankEncoder


class PropbankTest(unittest.TestCase):

    def setUp(self):
        tests_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
        root_dir = os.getcwd()
        mock_dir = '{:}mocks/'.format(tests_dir)

        gs_path = '{:}gs.json'.format(mock_dir)
        self._initialize_gs(gs_path)

        word2vec_path = '{:}word2vec.json'.format(mock_dir)
        self._initialize_word2vec(word2vec_path)

        schema_path = '{:}/{:}gs.yaml'.format(root_dir, config.SCHEMA_DIR)
        self._initialize_schema(schema_path)

        self.propbank_encoder = \
            PropbankEncoder(self.gs_dict, self.schema_dict, word2vec=self.word2vec_dict)

    def test_value(self):
        print(self.propbank_encoder.db)
        self.assertEqual(True, True)

    def _initialize_gs(self, gs_path):
        with open(gs_path, mode='r') as f:
            gs_json = json.load(f)

        self.gs_dict = {
            keycol_: {int(key_): val_ for key_, val_ in innerdict_.items()}
            for keycol_, innerdict_ in gs_json.items()}

    def _initialize_word2vec(self, wv_path):
        with open(wv_path, mode='r') as f:
            json_dict = json.load(f)

        self.word2vec_dict = {w: np.array(v) for w, v in json_dict.items()}

    def _initialize_schema(self, schema_path):
        with open(schema_path, mode='r') as f:
            self.schema_dict = yaml.load(f)
