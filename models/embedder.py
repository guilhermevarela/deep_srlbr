'''
    Created on seo 12, 2018

    @author: Varela

    Embedders provides vertorial representations
'''

import tensorflow as tf
from models.properties import lazy_property


class Embedder(object):
    def __init__(self, X, W2V, L, indices, trainable=True):
        '''Embedder defines the trainable embedding layer

        Defines the computing graph embedding phase

        Arguments:
            X {tf.placeholder} -- Placeholder for inputs
                zero padded tensor having [BATCH,MAX_TIME,FEATURE]

            W2V {tf.Variable} -- Pre Trained word embeddings

            L {tf.placeholder} -- list of batch size having max time

            indices {list} -- list with the indices from features to embedd

        Keyword Arguments:
            trainable {bool} -- if false will hold (default: {True})
        '''

        self.X = X
        self.W2V = W2V
        self.vocab_sz, self.embeddings_sz = W2V.get_shape()
        self.batch = L.get_shape()[0]
        self.L = L
        self.indices = indices
        self.trainable = trainable


        with tf.variable_scope('embedding_op'):
            self.rng = tf.range(self.batch, delta=1)
            self.slice
            self.lookup

    @lazy_property
    def lookup(self):
        def lookup_step(_, i):
            length = tf.gather(self.L, 0)
            features = self.Xs.get_shape()[-1]
            new_shape = (length, -1)

            begin_ = (i, 0, 0)
            shape_ = (1, length, features)
            slice_ids = tf.slice(self.Xs, begin_, shape_)
            slice_ids = tf.squeeze(slice_ids, axis=0)

            lookup = tf.nn.embedding_lookup(self.W2V, slice_ids)
            lookup = tf.reshape(lookup, new_shape)

            return lookup

        with tf.variable_scope('lookup'):
            batch, time, features = self.X.get_shape()
            self.Xe = tf.scan(
                lambda a, i: lookup_step(a, i),
                self.rng,
                initializer=lookup_step(0, 0)
            )
        return self.Xe

    @lazy_property
    def slice(self):
        with tf.variable_scope('slice'):
            #  Get bounds for the tensor
            batch, time, features = self.X.get_shape()

            slice_shape = [batch, time, 1]
            slice_list = []
            compl_list = []
            for idx in range(features):
                s = tf.slice(self.X, [0, 0, idx], slice_shape)
                if idx in self.indices:
                    slice_list.append(s)
                else:
                    compl_list.append(s)

            self.Xs = tf.concat(slice_list, 2)
            self.Xc = tf.concat(compl_list, 2)
        return self.Xs, self.Xc
