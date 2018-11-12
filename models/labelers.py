'''
    Created on May 31, 2018

    @author: Varela

    refs:
        https://danijar.com/structuring-your-tensorflow-models/
'''
import tensorflow as tf

from models.lib.properties import delegate_property, lazy_property
from models.propagators import get_propagator

from models.predictors import CRFPredictor

class LabelerMeta(type):
    '''This is a metaclass -- enforces method definition
    on function body

    Every labeler must implent the following methods
    * propagate -- builds automatic features from input layer
    * cost -- sum of the cost for each task (Predictor)
    * predict -- performs assignment of tags
    * label -- performs cost optmization over tags
    * error -- computes accuracy for each task

    References:
        https://docs.python.org/3/reference/datamodel.html#metaclasses
        https://realpython.com/python-metaclasses/
    '''
    def __new__(meta, name, base, body):
        labeler_methods = ('propagate', 'cost', 'predict', 'label')

        for lm in labeler_methods:
            if lm not in body:
                msg = 'Labeler must implement {:}'.format(lm)
                raise TypeError(msg)

        return super().__new__(meta, name, base, body)


class Labeler(object, metaclass=LabelerMeta):
    def __init__(self, X, T, L,
                 learning_rate=5 * 1e-3, hidden_size=[32, 32], targets_size=[60],
                 rec_unit='BasicLSTM', stack='DB'):
        '''Sets the computation graph parameters

        Responsable for building computation graph

        Extends:
            metaclass=LabelerMeta

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

            targets_size {int} -- Parameter holding the layer sizes (default: {60})

        References:
            https://www.tensorflow.org/programmers_guide/graphs
        '''
        self.X = X
        self.T = T
        self.L = L

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.targets_size = targets_size

        propagator_cls = get_propagator(stack)
        self.propagator = propagator_cls(X, L, hidden_size, rec_unit=rec_unit)
        self.predictor = CRFPredictor(self.propagator.propagate, T, L)

        self.propagate
        self.cost
        self.label
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
    def label(self):
        '''Optimize

        [description]

        Decorators:
            lazy_property

        Returns:
            [type] -- [description]
        '''
        with tf.variable_scope('label'):
            kwargs = {'learning_rate': self.learning_rate}
            opt = tf.train.AdamOptimizer(**kwargs).minimize(self.cost)
        return opt


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


class DualLabeler(object, metaclass=LabelerMeta):

    def __init__(self, X, T, L, recon_depth=-1,
                 learning_rate=5 * 1e-3, hidden_size=[32, 32], targets_size=[60],
                 rec_unit='BasicLSTM'):
        '''Sets the computation graph parameters

        Responsable for building computation graph -- expects R and Y to be 
        stacked over fourth dimention of T

                              ____________
        type A:               |           |
             ___________      |    CRF    |------- Rhat  
             |         |     /|___________|
        X __ |  DBRNN  | __ / 
             |         |    \  ____________ 
             -----------     \|            |
              (center)        |    CRF     | ----- Yhat
                              |____________|


        type B:

                     ___________      
                     |         | Rhat 
              ______ |  CRF    | __ 
            /        |         |    \ 
           /         -----------     \
          /             (a)          \                      _____________
         /                             \                     |           |
        X                                (Ztilde, Rhat) ---- |     CRF   | --- Yhat
         \                              /                    |___________|
          \                            /                     
           \         ___________      /
            \        |         |     /
             \______ |  DBRNN  | __ / 
                     |         |    
                     -----------   Ztilde  
                       (down)

        type C:

                   _____________
                   |           | Rhat 
             ______|  CRF      | __ 
            /      |___________|    \ 
           /                         \ 
          /             (up)          \                     _____________      ___________
         /                             \                    |           |     |           |
        X                               (Ztilde, Rhat) ---- |  RBRNN    |-----|     CRF   | --- Yhat
         \                              /                   |___________|     |___________|
          \                            /
           \         ___________      /
            \        |         |     /
             \_______|  DBRNN  |____/ 
                     |         |    
                     -----------   Ztilde



        type D:

                   _____________      ___________      
                   |           |      |         | Rhat 
             ______|   DBRNN   |----- |  CRF    | __ 
            /      |___________|      |         |    \ 
           /                          -----------     \
          /             (up)                           \                     _____________
         /                                              \                    |           |
        X                                                (Ztilde, Rhat) ---- |     CRF   | --- Yhat
         \                                               /                   |___________|
          \                                             / 
           \                ___________                /
            \               |         |               /
             \_____________ |  DBRNN  | _____________/ 
                            |         |    
                            -----------   Ztilde  
                                (down)


        type E:

                   _____________     ___________      
                   |           |     |         | Rhat 
             ______|   DBRNN   |-----|  CRF    | __ 
            /      |___________|     |_________|    \ 
           /                                         \ 
          /             (up)                          \                     _____________      ___________
         /                                             \                    |           |     |           |
        X                                               (Ztilde, Rhat) ---- |  RBRNN    |-----|     CRF   | --- Yhat
         \                                               /                  |___________|     |___________|
          \                                             /
           \                    ___________            /
            \                   |         |           /
             \________________  |  DBRNN  | _________/ 
                                |         |    
                                -----------   Ztilde  
                                  (down)         

        
        Extends:
            metaclass=LabelerMeta


        Arguments:
            X {object<tf.placeholder>} -- A 3D float tensor in which the dimensions are
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * features -- features dimension

            T {object<tf.placeholder>}  -- A 4D int32 tensor in which the dimensions are
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * features -- target IOB dimensions
                * 2        --- first dimension logits from recognition task (R)
                               second dimension logits from SRL task in IOB format (Y)
                               first dimension will be zero padded

            L {list<int>} -- a python list of integers carrying the sizes of 
                each proposition


        Keyword Arguments:
            learning_rate {float} -- Parameter to be used during optimization (default: {5 * 1e-3})
            hidden_size {list<int>} --  Parameter holding the layer sizes (default: {`[32, 32]`})
            targets_size {int} -- Parameter holding the layer sizes (default: {60})

        References:
            https://www.tensorflow.org/programmers_guide/graphs
        '''
        self.X = X
        self.recon_depth = recon_depth

        R, Y = tf.split(T, 2, axis=3)

        # Remove extra-axis
        R = tf.squeeze(R, axis=3)
        Y = tf.squeeze(Y, axis=3)

        # Remove paddings
        shape0 = (-1, -1, targets_size[0])
        self.R = tf.slice(R, begin=[0, 0, 0], size=shape0)

        shape1 = (-1, -1, targets_size[1])
        self.Y = tf.slice(Y, begin=[0, 0, 0], size=shape1)

        self.L = L

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.targets_size = targets_size

        self.propagators = []
        self.predictors = []
        if recon_depth == -1:  # TYPE A
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size, rec_unit=rec_unit, scope_label='center')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.R, L, scope_label='R')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.Y, L, scope_label='Y')
            )
        elif recon_depth == 1 and len(hidden_size) == 1: # TYPE B

            self.predictors.append(
                CRFPredictor(X, self.R, L, scope_label='R')
            )
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth], rec_unit=rec_unit, scope_label='down')
            )

            self.predictors.append(
                CRFPredictor(self.concat_op, self.Y, self.L, scope_label='Y')
            )

        elif recon_depth == 1 and len(hidden_size) > 1: # TYPE C
            self.predictors.append(
                CRFPredictor(X, self.R, L, scope_label='R')
            )
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth], rec_unit=rec_unit, scope_label='down')
            )
            self.propagators.append(
                InterleavedPropagator(self.concat_op, L, hidden_size[:recon_depth], rec_unit=rec_unit, scope_label='center')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.Y, L, scope_label='Y')
            )

        elif recon_depth >= 2 and recon_depth == len(hidden_size): # TYPE D
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth], rec_unit=rec_unit, scope_label='down')
            )
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth-1], rec_unit=rec_unit, scope_label='up')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.R, L, scope_label='R')
            )
            self.predictors.append(
                CRFPredictor(self.concat_op, self.Y, L, scope_label='Y')
            )
        elif recon_depth >= 2 and recon_depth < len(hidden_size): # TYPE E
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth], rec_unit=rec_unit, scope_label='down')
            )
            self.propagators.append(
                InterleavedPropagator(X, L, hidden_size[:recon_depth-1], rec_unit=rec_unit, scope_label='up')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.R, L, scope_label='R')
            )
            self.propagators.append(
                InterleavedPropagator(self.concat_op, L, hidden_size[recon_depth:], rec_unit=rec_unit, scope_label='center')
            )
            self.predictors.append(
                CRFPredictor(self.propagators[-1].propagate, self.Y, L, scope_label='Y')
            )


        else:
            raise NotImplementedError('This combination of parameters is not implemented')

        self.propagate
        self.cost
        self.label
        self.predict
        self.error

    # Delegates to Propagator
    # @delegate_property
    @lazy_property
    def propagate(self):
        res = None
        for p in self.propagators:
            res = p.propagate
        return res

    # Delegates to predictor
    # @delegate_property
    @lazy_property
    def cost(self):
        return self.predictors[0].cost + self.predictors[-1].cost

    # @delegate_property
    @lazy_property
    def predict(self):
        Rh = self.predictors[0].predict
        Yh = self.predictors[-1].predict
        return (Rh, Yh)

    @lazy_property
    def label(self):
        '''Optimize

        [description]

        Decorators:
            lazy_property

        Returns:
            [type] -- [description]
        '''
        with tf.variable_scope('label'):
            kwargs = {'learning_rate': self.learning_rate}
            optimum = tf.train.AdamOptimizer(**kwargs).minimize(self.cost)
        return optimum

    @lazy_property
    def error(self):
        Rh, Yh = self.predict
        mr, nr = self._abserror(Rh, self.R)
        my, ny = self._abserror(Yh, self.Y)

        errors = (mr + my) / (nr + ny)
        return errors

    def _abserror(self, Y, T):
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

        return tf.reduce_sum(mistakes), tf.reduce_sum(mask)

    @lazy_property
    def concat_op(self):
        Rhat = tf.one_hot(
            self.predictors[0].predict,
            3,
            on_value=1,
            off_value=0
        )
        Ztilde = self.propagators[0].propagate

        return tf.concat((Ztilde, tf.cast(Rhat, tf.float32)), axis=2)
