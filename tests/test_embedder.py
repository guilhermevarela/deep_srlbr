'''Tests Embedder

    Created on Sep 11, 2018

    @author: Varela
'''
import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal
import tensorflow as tf

from models.propbank_encoder import PropbankEncoder
from models.embedder import Embedder
import utils.corpus as cps

TESTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
FIX_DIR = '{:}fixtures/'.format(TESTS_DIR)
FIX_EVALUATOR_DIR = '{:}evaluator_conll/'.format(FIX_DIR)
FIX_EVALUATOR_MOCK_DIR = '{:}mocks/'.format(FIX_EVALUATOR_DIR)
FIX_PROPBANK_DIR = '{:}propbank_encoder/'.format(FIX_DIR)
FIX_PROPBANK_PATH = '{:}deep_glo50.pickle'.format(FIX_PROPBANK_DIR)

FIX_TMP_DIR = '{:}lr1.00e-03_hs32x32_ctx-p1/'.format(FIX_EVALUATOR_DIR)


class EmbedderBaseTestCase(unittest.TestCase):

    def setUp(self):
        cps.get_db_bounds = unittest.mock.Mock(return_value=(1690, 1691))
        self.pe = PropbankEncoder.recover(FIX_PROPBANK_PATH)
        self.w2v = [np.array(emb_) for emb_ in self.pe.embeddings]
        self.w2v = np.array(self.w2v)

        # All model features to be embedded
        self.feature_list = ['ID', 'FORM', 'FORM_CTX_P-1',
                             'FORM_CTX_P+0', 'FORM_CTX_P+1',
                             'MARKER', 'GPOS']

        # Indices that are textual -- `FORM*` fields
        self.indices = [1, 2, 3, 4]

        X_list = []
        X_s = []  # X to slice
        X_c = []  # X complement to slice
        for i, feat_ in enumerate(self.feature_list):
            dict_ = self.pe.column('train', feat_, encoding='MIX')

            values_ = list(dict_.values())
            # This are the columns with len one
            if feat_ in ('ID', 'MARKER') or 'FORM' in feat_:
                npyval_ = np.array(values_, dtype=np.int32)
                npyval_ = npyval_.reshape((len(values_), 1))
            else:
                # This are the columns with lists of lists
                values_ = [np.array(fv_) for fv_ in values_]
                npyval_ = np.array(values_, dtype=np.int32)

            X_list.append(npyval_)
            if i in self.indices:
                X_s.append(npyval_)
            else:
                X_c.append(npyval_)

        # Make a two dim matrix  TIME X FEATURES
        self.X = np.concatenate(X_list, axis=1)
        # Convert to tree dim matrix BATCH x TIME X FEATURES
        self.X = np.expand_dims(self.X, 0)

        self.seqlen = [self.X.shape[1]]

        self.X_s = np.concatenate(X_s, axis=1)

        # Convert to tree dim matrix BATCH x TIME X FEATURES
        self.X_s = np.expand_dims(self.X_s, 0)

        self.X_c = np.concatenate(X_c, axis=1)

        # Convert to tree dim matrix BATCH x TIME X FEATURES
        self.X_c = np.expand_dims(self.X_c, 0)


class EmbedderSliceTestCase(EmbedderBaseTestCase):

    def test(self):
        # Presets
        X = tf.placeholder(dtype=tf.int32, shape=self.X.shape, name='X')

        W2V = tf.Variable(self.w2v, dtype=tf.int32, trainable=False)

        seqlen = tf.placeholder(dtype=tf.int32, shape=(1,), name='seqlen')
        # builds the computation graph
        self.embedder = Embedder(X, W2V, seqlen, self.indices)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as sess:
            X_s, X_c = sess.run(
                self.embedder._slice,
                feed_dict={X: self.X}
            )

        assert_array_equal(self.X_s, X_s)
        assert_array_equal(self.X_c, X_c)


class EmbedderW2VTestCase(EmbedderBaseTestCase):

    def test(self):
        # Presets
        X = tf.placeholder(dtype=tf.int32, shape=self.X.shape, name='X')

        W2V = tf.Variable(self.w2v, dtype=tf.int32, trainable=False)

        seqlen = tf.placeholder(dtype=tf.int32, shape=(1,), name='seqlen')
        # builds the computation graph
        self.embedder = Embedder(X, W2V, seqlen, self.indices)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as sess:
            sess.run(init_op)
            w2v = sess.run(
                self.embedder.W2V,
                feed_dict={X: self.X}
            )

        assert_array_equal(self.w2v, w2v)


class EmbedderLookupTestCase(EmbedderBaseTestCase):
    def setUp(self):
        super(EmbedderLookupTestCase, self).setUp()

        text_features = [f
                         for i, f in enumerate(self.feature_list)
                         if i in self.indices]

        feature_iter = self.pe.iterator(
            'train',
            filter_columns=text_features,
            encoding='EMB'
        )

        X_list = []
        for idx, dict_ in feature_iter:
            idx_list = []
            for _, v in dict_.items():
                v_ = np.array(v, dtype=np.int32)
                v_ = np.expand_dims(v_, axis=1)
                idx_list.append(v_)

            X_list.append(np.concatenate(idx_list, axis=0))

        self.X_e = np.transpose(np.concatenate(X_list, axis=1))


    def test(self):
        # Presets
        X = tf.placeholder(dtype=tf.int32, shape=self.X.shape, name='X')

        W2V = tf.Variable(self.w2v, dtype=tf.int32, trainable=False)

        L = tf.placeholder(dtype=tf.int32, shape=(1,), name='L')
        # builds the computation graph
        self.embedder = Embedder(X, W2V, L, self.indices)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as sess:
            sess.run(init_op)

            X_e = sess.run(
                self.embedder._lookup,
                feed_dict={X: self.X, L: self.seqlen}
            )

        assert_array_equal(self.X_e, X_e[0, :, :])


class EmbedderEmbedTestCase(EmbedderLookupTestCase):
    def setUp(self):
        super(EmbedderEmbedTestCase, self).setUp()
        self.X_e = np.expand_dims(self.X_e, 0)
        self.V = np.concatenate((self.X_e, self.X_s), axis=2)

    def test(self):
        # Presets
        X = tf.placeholder(dtype=tf.int32, shape=self.X.shape, name='X')

        W2V = tf.Variable(self.w2v, dtype=tf.int32, trainable=False)

        L = tf.placeholder(dtype=tf.int32, shape=(1,), name='L')
        # builds the computation graph
        self.embedder = Embedder(X, W2V, L, self.indices)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as sess:
            sess.run(init_op)

            V = sess.run(
                self.embedder._merge,
                feed_dict={X: self.X, L: self.seqlen}
            )

        assert_array_equal(self.V, V)
