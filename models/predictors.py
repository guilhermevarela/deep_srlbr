'''Predictors compose the last layer of the DeepSrl System

    * Maps features to targets
    * Must evaluate cost
    * Gradient descend
'''
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
try:  # tensorflow 1.10
    from tensorflow.contrib.crf import crf_decode
except ImportError:
    # tensorflow 1.2.1
    from models.lib.crf import crf_decode

from models.lib.properties import lazy_property


class Predictor(object):

    def __init__(self, V, T, L, i=0):
        scope_id = 'CRF-{:}'.format(i)

        self.V = V
        self.T = T
        self.L = L
        self.i = i
        self.Tflat = tf.cast(tf.argmax(T, 2), tf.int32)

        with tf.variable_scope(scope_id):
            self.score
            self.cost
            self.predict

    def score(self):
        raise NotImplementedError('Predictor must implement cost')

    def cost(self):
        raise NotImplementedError('Predictor must implement cost')

    def predict(self):
        raise NotImplementedError('Predictor must implement predict')


class CRFPredictor(Predictor):
    @lazy_property
    def score(self):
        scope_id = 'score{:}'.format(self.i)
        with tf.variable_scope(scope_id):
            Wo = tf.Variable(tf.random_normal(self.wo_shape, name='Wo'))
            bo = tf.Variable(tf.random_normal(self.bo_shape, name='bo'))

            # Stacking is cleaner and faster - but it's hard to use for multiple pipelines
            self.S = tf.scan(
                lambda a, x: tf.matmul(x, Wo),
                self.V, initializer=tf.matmul(self.V[0], Wo)) + bo

        return self.S

    @lazy_property
    def cost(self):
        '''Computes the viterbi_score after the propagation step, returns the cost.

        Consumes the representation coming from propagation layer, evaluates 
            the log_likelihod and parameters

        Decorators:
            lazy_property

        Returns:
            cost {tf.float64} -- A scalar holding the average log_likelihood 
            of the loss by estimatiof
        '''
        scope_id = 'cost{:}'.format(self.i)
        with tf.variable_scope(scope_id):
            args = (self.S, self.Tflat, self.L)
            log_likelihood, self.transition_params = crf_log_likelihood(*args)

        return tf.reduce_mean(-log_likelihood)

    @lazy_property
    def predict(self):
        '''Decodes the viterbi score for the inputs

        Consumes both results from propagation and and cost layers

        Decorators:
            lazy_property

        Returns:
            [type] -- [description]
        '''
        scope_id = 'prediction{:}'.format(self.i)
        with tf.variable_scope(scope_id):
            # Compute the viterbi sequence and score.
            args = (self.S, self.transition_params, self.L)
            viterbi_sequence, viterbi_score = crf_decode(*args)

        return tf.cast(viterbi_sequence, tf.int32)

    @lazy_property
    def wo_shape(self):
        # Static dimensions are of class Dimension(n)
        v = int(self.V.get_shape()[-1])
        t = int(self.T.get_shape()[-1])
        return (v, t)

    @lazy_property
    def bo_shape(self):
        # Static dimensions are of class Dimension(n)
        t = int(self.T.get_shape()[-1])
        return (t,)
