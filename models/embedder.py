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
        self.W2V = W2V
        self.seqlen = seqlen
        self.indices = indices
        self.trainable = trainable

        self.slice

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
