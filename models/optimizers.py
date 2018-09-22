'''
    Created on May 31, 2018

    @author: Varela

    refs:
        https://danijar.com/structuring-your-tensorflow-models/
'''
import tensorflow as tf

from models.properties import delegate_property, lazy_property
from models.propagators import InterleavedPropagator

from models.predictors import CRFPredictor


class Optmizer(object):
    def __init__(self, X, T, seqlens,
                 learning_rate=5 * 1e-3, hidden_size=[32, 32], target_sz_list=[60],
                 ru='BasicLSTM'):
        '''Sets the computation graph parameters

        Responsable for building computation graph

        refs:
            https://www.tensorflow.org/programmers_guide/graphs

        Arguments:
            X {object<tf.placeholder>} -- A 3D float tensor in which the dimensions are
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * features -- features dimension

            T {object<tf.placeholder>}  -- A 3D float tensor in which the dimensions are
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * features -- features dimension

            seqlens {list<int>} -- a python list of integers carrying the sizes of 
                each proposition


        Keyword Arguments:
            learning_rate {float} -- Parameter to be used during optimization (default: {5 * 1e-3})
            hidden_size {list<int>} --  Parameter holding the layer sizes (default: {`[32, 32]`})
            target_sz_list {int} -- Parameter holding the layer sizes (default: {60})
        '''
        self.optmizer = 'DBLSTM'
        self.X = X
        self.T = T
        self.seqlens = seqlens

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.target_sz_list = target_sz_list

        self.propagator = InterleavedPropagator(
            X, seqlens, hidden_size, ru=ru)
        self.predictor = CRFPredictor(self.propagator.propagate, T, seqlens)

        self.propagate
        self.cost
        self.optimize
        self.predict
        self.error

    # Delegates to Propagator
    @delegate_property
    def propagate(self):
        pass
    # def propagate(self):
    #     return self.propagator.propagate

    # Delegates to predictor
    @delegate_property
    def cost(self):
        pass

    # def cost(self):
    #     return self.predictor.cost
    @delegate_property
    def predict(self):
        pass

    # def predict(self):
    #     return self.predictor.predict

    @lazy_property
    def optimize(self):
        '''Optimize

        [description]

        Decorators:
            lazy_property

        Returns:
            [type] -- [description]
        '''
        with tf.variable_scope('optimize'):
            kwargs = {'learning_rate': self.learning_rate}
            optimum = tf.train.AdamOptimizer(**kwargs).minimize(self.cost)
        return optimum


    @lazy_property
    def error(self):
        '''Computes the prediction errors

        Compares target tags to predicted tags

        Decorators:
            lazy_property

        Returns:
            error {float} -- percentage of wrong tokens
        '''
        mistakes = tf.not_equal(self.predict, tf.cast(tf.argmax(self.T, 2), tf.int32))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.T), reduction_indices=2))
        mask = tf.cast(mask, tf.float32)
        mistakes *= mask

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.seqlens, tf.float32)
        return tf.reduce_mean(mistakes)
