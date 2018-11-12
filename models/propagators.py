'''Propagartors generate intermediate features 
    which are relevant to solve sequence modeling

    * Maps features into features
    * Composed of cell units
'''
import tensorflow as tf
from models.lib.properties import lazy_property

def get_unit(sz, rec_unit='BasicLSTM'):
    ru_types = ('BasicLSTM', 'GRU', 'LSTM', 'LSTMBlockCell')
    if rec_unit not in ru_types:
        raise ValueError('recurrent_unit {:} must be in {:}'.format (rec_unit, ru_types))

    if rec_unit == 'BasicLSTM':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(sz,
                                                forget_bias=1.0,
                                                state_is_tuple=True)
    if rec_unit == 'GRU':
        rnn_cell = tf.nn.rnn_cell.GRUCell(sz)

    if rec_unit == 'LSTM':
        rnn_cell = tf.nn.rnn_cell.LSTMCell(sz,
                                           use_peepholes=True,
                                           forget_bias=1.0,
                                           state_is_tuple=True)

    # Should run faster then BasicLSTM and LSTM
    if rec_unit == 'LSTMBlockCell':
        rnn_cell = tf.nn.rnn_cell.LSTMBlockCell(sz,
                                           use_peepholes=True,
                                           forget_bias=1.0,
                                           state_is_tuple=True)
    return rnn_cell

def get_propagator(stack='DB'):
    if stack == 'DB':
        return InterleavedPropagator

    if stack == 'BI':
        return BiPropagator

    raise ValueError(f'stack must be in (`BI`, `DB`) got {stack}')


class PropagatorMeta(type):
    '''This is a metaclass -- enforces method definition
    on function body

    Every Propagator must implent the following methods
    * propagate - builds automatic features from input layer

    References:
        https://docs.python.org/3/reference/datamodel.html#metaclasses
        https://realpython.com/python-metaclasses/
    '''
    def __new__(meta, name, base, body):
        propagator_methods = ('propagate',)

        for pm in propagator_methods:
            if pm not in body:
                msg = 'Agent must implement {:}'.format(pm)
                raise TypeError(msg)

        return super().__new__(meta, name, base, body)

class BasePropagator(object):

    def __init__(self, V, L, hidden_layers, rec_unit='BasicLSTMCell', scope_label=''):
        '''Builds a recurrent neural network section of the graph

        Extends:
            metaclass=PropagatorMeta

        Arguments:
            V {tensor} -- Rank 3 input tensor having dimensions as follows
                        [batches, max_time, features]

            L {tensor} -- Rank 1 length tensor having the true length of each example

            hidden_layers {list} -- Hidden list

        Keyword Arguments:
            rec_unit {str} -- Name of the recurrent unit (default: {'BasicLSTMCell'})
            scope_label {str} -- Label for this scope (default: {''})

        References:
            Jie Zhou and Wei Xu. 2015.

            "End-to-end learning of semantic role labeling using recurrent neural
            networks". In Proc. of the Annual Meeting of the Association
            for Computational Linguistics (ACL)

            http://www.aclweb.org/anthology/P15-1109
        '''
        # self.scope_id = '{:}-{:}'.format(rec_unit, scope_label)

        self.V = V
        self.L = L

        self.rec_unit = rec_unit
        self.scope_label = scope_label

        self.hidden_layers = hidden_layers
        with tf.variable_scope(self.scope_id):
            self.propagate


class InterleavedPropagator(BasePropagator, metaclass=PropagatorMeta):

    # def __init__(self, V, L, hidden_layers, rec_unit='BasicLSTMCell', scope_label=''):
    #     '''Builds a recurrent neural network section of the graph

    #     Extends:
    #         metaclass=PropagatorMeta

    #     Arguments:
    #         V {tensor} -- Rank 3 input tensor having dimensions as follows
    #                     [batches, max_time, features]

    #         L {tensor} -- Rank 1 length tensor having the true length of each example

    #         hidden_layers {list} -- Hidden list

    #     Keyword Arguments:
    #         rec_unit {str} -- Name of the recurrent unit (default: {'BasicLSTMCell'})
    #         scope_label {str} -- Label for this scope (default: {''})

    #     References:
    #         Jie Zhou and Wei Xu. 2015.

    #         "End-to-end learning of semantic role labeling using recurrent neural
    #         networks". In Proc. of the Annual Meeting of the Association
    #         for Computational Linguistics (ACL)

    #         http://www.aclweb.org/anthology/P15-1109
    #     '''

    #     scope_id = 'DB{:}{:}'.format(rec_unit, scope_label)

    #     self.V = V
    #     self.L = L

    #     self.rec_unit = rec_unit
    #     self.scope_label = scope_label

    #     self.hidden_layers = hidden_layers
    #     with tf.variable_scope(scope_id):
    #         self.propagate
    def __init__(self, V, L, hidden_layers, rec_unit='BasicLSTMCell', scope_label=''):
        self.scope_id = f'DB{rec_unit}_{scope_label}'
        super(InterleavedPropagator, self).__init__(V, L, hidden_layers, rec_unit=rec_unit, scope_label=scope_label)

    @lazy_property
    def propagate(self):
        '''Forward propagates the inputs V thru interlaced bi-lstm network

        The inputs X are evaluated by each hidden layer (forward propagating)
        resulting in scores to be consumed by prediction layer

        Decorators:
            lazy_property

        Returns:
            score {tf.Variable} -- a 3D float tensor in which
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * target_sz_list -- ouputs dimension
        '''
        with tf.variable_scope('fw0'):
            self.cell_fw = get_unit(self.hidden_layers[0], self.rec_unit)

            outputs_fw, states = tf.nn.dynamic_rnn(
                cell=self.cell_fw,
                inputs=self.V,
                sequence_length=self.L,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('bw0'):
            self.cell_bw = get_unit(self.hidden_layers[0], self.rec_unit)

            inputs_bw = tf.reverse_sequence(
                outputs_fw,
                self.L,
                batch_axis=0,
                seq_axis=1
            )

            outputs_bw, states = tf.nn.dynamic_rnn(
                cell=self.cell_bw,
                inputs=inputs_bw,
                sequence_length=self.L,
                dtype=tf.float32,
                time_major=False
            )
            outputs_bw = tf.reverse_sequence(
                outputs_bw,
                self.L,
                batch_axis=0,
                seq_axis=1
            )

        h = outputs_bw
        h_1 = outputs_fw
        for i, sz in enumerate(self.hidden_layers[1:]):

            with tf.variable_scope('fw{:}'.format(i + 1)):
                inputs_fw = tf.concat((h, h_1), axis=2)
                self.cell_fw = get_unit(sz, self.rec_unit)

                outputs_fw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_fw,
                    inputs=inputs_fw,
                    sequence_length=self.L,
                    dtype=tf.float32,
                    time_major=False)

            with tf.variable_scope('bw{:}'.format(i + 1)):
                inputs_bw = tf.concat((outputs_fw, h), axis=2)
                inputs_bw = tf.reverse_sequence(
                    inputs_bw,
                    self.L,
                    batch_axis=0,
                    seq_axis=1
                )
                self.cell_bw = get_unit(sz, self.rec_unit)

                outputs_bw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_bw,
                    inputs=inputs_bw,
                    sequence_length=self.L,
                    dtype=tf.float32,
                    time_major=False)

                outputs_bw = tf.reverse_sequence(
                    outputs_bw,
                    self.L,
                    batch_axis=0,
                    seq_axis=1
                )

            h = outputs_bw
            h_1 = outputs_fw

        # self.V = tf.concat((h, h_1), axis=2)

        # return self.V
        return tf.concat((h, h_1), axis=2)

class BiPropagator(BasePropagator, metaclass=PropagatorMeta):

    def __init__(self, V, L, hidden_layers, rec_unit='BasicLSTMCell', scope_label=''):
        self.scope_id = f'Bi{rec_unit}_{scope_label}'
        super(BiPropagator, self).__init__(V, L, hidden_layers, rec_unit=rec_unit, scope_label=scope_label)

    @lazy_property
    def propagate(self):
        '''Forward propagates the inputs V thru interlaced bi-lstm network

        The inputs X are evaluated by each hidden layer (forward propagating)
        resulting in scores to be consumed by prediction layer

        Decorators:
            lazy_property

        Returns:
            score {tf.Variable} -- a 3D float tensor in which
                * batch_size -- fixed sample size from examples
                * max_time -- maximum time from batch_size examples (default: None)
                * target_sz_list -- ouputs dimension
        '''

        inputs = self.V
        for i, h in enumerate(self.hidden_layers):
            with tf.variable_scope(f'h{i}'):
                fw = get_unit(h, rec_unit=self.rec_unit)
                bw = get_unit(h, rec_unit=self.rec_unit)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    fw, bw, inputs,
                    sequence_length=self.L,
                    dtype=tf.float32,
                    time_major=False,
                )
                inputs = tf.concat(outputs, axis=2)

        return inputs
