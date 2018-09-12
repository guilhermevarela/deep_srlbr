'''
    Created on seo 12, 2018

    @author: Varela

    Embedders provides vertorial representations
'''

import tensorflow as tf
from models.properties import lazy_property


class Embedder(object):
    def __init__(self, X, W2V, seqlen, indices, trainable=True):
        '''Embedder defines the trainable embedding layer

        Defines the computing graph embedding phase

        Arguments:
            X {tf.placeholder} -- Placeholder for inputs
                zero padded tensor having [BATCH,MAX_TIME,FEATURE]

            W2V {np.array} -- Pre Trained word embeddings

            seqlen {list} -- list of batch size having max time

            indices {list} -- list with the indices from features to embedd

        Keyword Arguments:
            trainable {bool} -- if false will hold (default: {True})
        '''

        self.X = X
        self.vocab_sz, self.embeddings_sz = W2V.shape
        self.seqlen = seqlen
        self.indices = indices
        self.trainable = trainable

        with tf.variable_scope('embedding_op'):
            self.W2V = tf.Variable(W2V, trainable=trainable)
            self.slice
            self.lookup

    @lazy_property
    def lookup(self):
        def lookup_init():
            begin_ = (0, 0, 0)
            shape_ = (0, tf.gather(self.seqlen, 0), 1)
            slice_ids = tf.slice(self.X, begin_, shape_)
            return tf.nn.embedding_lookup(self.W2V, slice_ids)

        def lookup_step(a, i):
            begin_ = (i, 0, 0)
            shape_ = (0, tf.gather(self.seqlen, i), 1)
            slice_ids = tf.slice(self.X, begin_, shape_)
            return tf.nn.embedding_lookup(self.W2V, slice_ids)

        with tf.variable_scope('lookup'):
            batch, time, features = self.X.get_shape()
            self.rng = tf.range(batch, delta=1)
            self.Xe = tf.scan(
                lambda a, i: lookup_step(a, i),
                self.rng,
                initializer=lookup_init()
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
