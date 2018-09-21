'''Predictors compose the last layer of the DeepSrl System

    * Maps features to targets
    * Must evaluate cost
    * Gradient descend
'''
import tensorflow as tf
from models.properties import lazy_property


class Predictor(object):

    def __init__(self, Scores, T, seqlens):
        self.predictor = 'CRF'

        self.Scores = Scores
        self.T = tf.cast(tf.argmax(T, 2), tf.int32)
        self.seqlens = seqlens

        self.cost
        self.predict

    def cost(self):
        raise NotImplementedError('Predictor must implement cost')

    def predict(self):
        raise NotImplementedError('Predictor must implement predict')


class CRFPredictor(Predictor):

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
        with tf.variable_scope('cost'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.Scores, self.T, self.seqlens)

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
        with tf.variable_scope('prediction'):
            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.Scores, self.transition_params, self.seqlens)

        return tf.cast(viterbi_sequence, tf.int32)


class CRFDualLabelPredictor(Predictor):
    '''Computes the viterbi_score for dual label tasks

    Previous works show that the argument recognition subtask is
    important. Have it being computed in parallel instead of
    having it computed as a pipeline

    Instead of having:
        B-A0, I-A0, B-V, B-A1, I-A1, I-A1

    Have:
        (B, A0), (I, A0), (B, V), (B, A1), (I, A1), (I, A1)

    Extends:
        Predictor
    '''
    def __init__(self, Scores, T, seqlens):
        self.predictor = 'CRF'

        self.Scores = Scores
        self.T = tf.cast(tf.argmax(T, 2), tf.int32)
        self.seqlens = seqlens

        self.cost
        self.predict

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
        with tf.variable_scope('cost'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.Scores, self.T, self.seqlens)

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
        with tf.variable_scope('prediction'):
            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.Scores, self.transition_params, self.seqlens)

        return tf.cast(viterbi_sequence, tf.int32)