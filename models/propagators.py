'''Propagartors generate intermediate features 
    which are relevant to solve sequence modeling

    * Maps features into features
    * Composed of cell units
'''
import tensorflow as tf
from models.properties import lazy_property


def get_unit(sz, ru='BasicLSTM'):
    ru_types = ('BasicLSTM', 'GRU', 'LSTM')
    if ru not in ru_types:
        raise ValueError('recurrent_unit {:} must be in {:}'.format (ru, ru_types))

    if ru == 'BasicLSTM':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(sz,
                                                forget_bias=1.0,
                                                state_is_tuple=True)
    if ru == 'GRU':
        rnn_cell = tf.nn.rnn_cell.GRUCell(sz)

    if ru == 'LSTM':
        rnn_cell = tf.nn.rnn_cell.LSTMCell(sz,
                                use_peepholes=False,
                                forget_bias=1.0,
                                state_is_tuple=True)
    return rnn_cell


class Propagator(object):

    def __init__(self, V, seqlens, hidden_sz_list, ru='BasicLSTMCell'):
        self.ru = ru
        self.propagator = 'interleaved'

        self.hidden_sz_list = hidden_sz_list
        self.V = V
        self.seqlens = seqlens

        self.propagate

    def propagate(self):
        raise NotImplementedError('Propagator must implement propagate')


class InterleavedPropagator(Propagator):

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
            self.cell_fw = get_unit(self.hidden_sz_list[0], self.ru)

            outputs_fw, states = tf.nn.dynamic_rnn(
                cell=self.cell_fw,
                inputs=self.V,
                sequence_length=self.seqlens,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('bw0'):
            self.cell_bw = get_unit(self.hidden_sz_list[0], self.ru)

            inputs_bw = tf.reverse_sequence(
                outputs_fw,
                self.seqlens,
                batch_axis=0,
                seq_axis=1
            )

            outputs_bw, states = tf.nn.dynamic_rnn(
                cell=self.cell_bw,
                inputs=inputs_bw,
                sequence_length=self.seqlens,
                dtype=tf.float32,
                time_major=False
            )
            outputs_bw = tf.reverse_sequence(
                outputs_bw,
                self.seqlens,
                batch_axis=0,
                seq_axis=1
            )

        h = outputs_bw
        h_1 = outputs_fw
        for i, sz in enumerate(self.hidden_sz_list[1:]):

            with tf.variable_scope('fw{:}'.format(i + 1)):
                inputs_fw = tf.concat((h, h_1), axis=2)
                self.cell_fw = get_unit(sz, self.ru)

                outputs_fw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_fw,
                    inputs=inputs_fw,
                    sequence_length=self.seqlens,
                    dtype=tf.float32,
                    time_major=False)

            with tf.variable_scope('bw{:}'.format(i + 1)):
                inputs_bw = tf.concat((outputs_fw, h), axis=2)
                inputs_bw = tf.reverse_sequence(
                    inputs_bw,
                    self.seqlens,
                    batch_axis=0,
                    seq_axis=1
                )
                self.cell_bw = get_unit(sz, self.ru)

                outputs_bw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_bw,
                    inputs=inputs_bw,
                    sequence_length=self.seqlens,
                    dtype=tf.float32,
                    time_major=False)

                outputs_bw = tf.reverse_sequence(
                    outputs_bw,
                    self.seqlens,
                    batch_axis=0,
                    seq_axis=1
                )

            h = outputs_bw
            h_1 = outputs_fw

        self.V = tf.concat((h, h_1), axis=2)

        return self.V

    @lazy_property
    def output_shape(self):
        return [None, None] + [2 * self.hidden_sz_list[-1]]
