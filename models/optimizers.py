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
    def __init__(self, X, T, L,
                 learning_rate=5 * 1e-3, hidden_size=[32, 32], target_sizes=[60],
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

            L {list<int>} -- a python list of integers carrying the sizes of 
                each proposition


        Keyword Arguments:
            learning_rate {float} -- Parameter to be used during optimization (default: {5 * 1e-3})
            hidden_size {list<int>} --  Parameter holding the layer sizes (default: {`[32, 32]`})
            target_sizes {int} -- Parameter holding the layer sizes (default: {60})
        '''
        self.optmizer = 'DBLSTM'
        self.X = X
        self.T = T
        self.L = L

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.target_sizes = target_sizes

        self.propagator = InterleavedPropagator(X, L, hidden_size, ru=ru)
        self.predictor = CRFPredictor(self.propagator.propagate, T, L)

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
        mistakes /= tf.cast(self.L, tf.float32)
        return tf.reduce_mean(mistakes)


class OptmizerDualLabel(object):
    def __init__(self, X, T, L, r_depth=-1,
                 learning_rate=5 * 1e-3, hidden_size=[32, 32], target_sizes=[60],
                 ru='BasicLSTM'):
        '''Sets the computation graph parameters

        Responsable for building computation graph -- expects dual targets T

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

            L {list<int>} -- a python list of integers carrying the sizes of 
                each proposition


        Keyword Arguments:
            learning_rate {float} -- Parameter to be used during optimization (default: {5 * 1e-3})
            hidden_size {list<int>} --  Parameter holding the layer sizes (default: {`[32, 32]`})
            target_sizes {int} -- Parameter holding the layer sizes (default: {60})
        '''
        self.optmizer = 'DBLSTM'
        self.X = X
        self.r_depth = r_depth

        R, C = tf.split(T, 2, axis=3)

        # Remove extra-axis
        R = tf.squeeze(R, axis=3)
        C = tf.squeeze(C, axis=3)

        # Remove paddings
        shape0 = (-1, -1, target_sizes[0])
        self.R = tf.slice(R, begin=[0, 0, 0], size=shape0)

        shape1 = (-1, -1, target_sizes[1])
        self.C = tf.slice(C, begin=[0, 0, 0], size=shape1)

        self.L = L

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.target_sizes = target_sizes

        if r_depth == -1 or r_depth == len(hidden_size):
            self.propagator_0 = InterleavedPropagator(X, L, hidden_size, ru=ru)

            self.predictor_0 = CRFPredictor(self.propagator_0.propagate, self.R, L, i=0)
            self.predictor_1 = CRFPredictor(self.propagator_0.propagate, self.C, L, i=1)

        elif r_depth == 1:
            # up branch --> Handles recognition task
            # [BATCH, MAX_TIME, TARGET_IOB] this tensor is zero padded from 3rd position
            self.predictor_0 = CRFPredictor(self.X, self.R, L, i=0)

            # down branch --> Handles classification
            # [BATCH, MAX_TIME, 2*hidden_size[:1]] this tensor is zero padded from 3rd position
            self.propagator_0 = InterleavedPropagator(X, L, hidden_size[:r_depth], ru=ru)

            # merge the represenations
            # print(self.predictor_0.get_shape())
            # print(self.propagator_0.get_shape())
            self.Xhat = tf.concat((self.propagator_0.propagate, tf.cast(self.predictor_0.predict, tf.float32)), axis=2)

            # joint propagator
            self.propagator_1 = InterleavedPropagator(self.Xhat, L, hidden_size[r_depth:], ru=ru)
            self.predictor_1 = CRFPredictor(self.propagator_1, self.C, L, i=1)

        else:
            raise NotImplementedError('This combination of parameters is not implemented')

        self.propagate
        self.cost
        self.optimize
        self.predict
        self.error

    # Delegates to Propagator
    # @delegate_property
    @lazy_property
    def propagate(self):
        if self.r_depth == -1 or self.r_depth == len(self.hidden_size):
            return self.propagator_0.propagate
        else:
            return self.propagator_1.propagate

    # Delegates to predictor
    # @delegate_property
    @lazy_property
    def cost(self):
        return self.predictor_0.cost + self.predictor_1.cost

    # @delegate_property
    @lazy_property
    def predict(self):
        Y0 = self.predictor_0.predict
        Y1 = self.predictor_1.predict
        return (Y0, Y1)

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
        errors = self._error(self.predictor_0.predict, self.R)
        errors += self._error(self.predictor_1.predict, self.C)
        return errors

    def _error(self, Y, T):
        '''Computes the prediction errors

        Compares target tags to predicted tags

        Decorators:
            lazy_property

        Returns:
            error {float} -- percentage of wrong tokens
        '''
        mistakes = tf.not_equal(Y, tf.cast(tf.argmax(T, 2), tf.int32))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(T), reduction_indices=2))
        mask = tf.cast(mask, tf.float32)
        mistakes *= mask

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.L, tf.float32)
        return tf.reduce_mean(mistakes)
