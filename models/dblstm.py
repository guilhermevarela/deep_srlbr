'''
    Created on May 31, 2018

    @author: Varela

    refs:
        https://danijar.com/structuring-your-tensorflow-models/
'''
import functools
import tensorflow as tf

from tf.nn.rnn_cell import BasicLSTMCell
from utils import error_rate2

from config import DATASET_TRAIN_GLO50_PATH


def get_unit(sz):
    return BasicLSTMCell(sz, forget_bias=1.0, state_is_tuple=True)


def lazy_property(function):
    '''
        It stores the result in a member named after the decorated function 
        (prepended with a prefix) and returns this value on any subsequent calls.        
    '''
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DataLayer(object):
    '''
      Defines the first layer for DBLSTM model
    '''
    def __init__(self, size):
        self.size = size
        with tf.variable_scope('fw'):
            self.cell_fw = get_unit(self.size)

        with tf.variable_scope('bw'):
            self.cell_bw = get_unit(self.size)

    @lazy_property
    def prediction(self, X, seqlens):
        with tf.variable_scope('fw{:}'.format(id)):
            # CREATE / REUSE FWD/BWD CELL
            self.cell_fw = get_unit(self.size)

            outputs_fw, states = tf.nn.dynamic_rnn(
                cell= self.cell_fw,
                inputs=X,
                sequence_length=seqlens,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('bw{:}'.format(id)):
            self.cell_bw = get_unit(self.size)

            inputs_bw = tf.reverse_sequence(outputs_fw, seqlens, batch_axis=0, seq_axis=1)

            outputs_bw, states = tf.nn.dynamic_rnn(
                cell=self.cell_bw,
                inputs=inputs_bw,
                sequence_length=seqlens,
                dtype=tf.float32,
                time_major=False
            )
            outputs_bw = tf.reverse_sequence(outputs_bw, seqlens, batch_axis=0, seq_axis=1)

        return outputs_bw, outputs_fw



class HiddenLayer(object):
    '''
      Defines the hidden layers for model
    '''
    def __init__(self, size):
        self.size = size

        with tf.variable_scope('fw{:}'.format(id)):
            self.cell_fw = get_unit(self.size)

        with tf.variable_scope('bw{:}'.format(id)):
            self.cell_bw = get_unit(self.size)

    @lazy_property
    def prediction(self, h, h_1, seqlens):
        with tf.variable_scope('fw{:}'.format(id)):
            inputs_fw = tf.concat((h, h_1), axis=2)

            self.cell_fw = get_unit(self.size)

            outputs_fw, states = tf.nn.dynamic_rnn(
                cell=self.cell_fw,
                inputs=inputs_fw,
                sequence_length=seqlens,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('bw'):
            inputs_bw = tf.concat((outputs_fw, h), axis=2)
            inputs_bw = tf.reverse_sequence(inputs_bw, seqlens, batch_axis=0, seq_axis=1)
            self.cell_bw = get_unit(self.size)

            outputs_bw, states = tf.nn.dynamic_rnn(
                cell=self.cell_bw,
                inputs=inputs_bw,
                sequence_length=seqlens,
                dtype=tf.float32,
                time_major=False
            )
            outputs_bw = tf.reverse_sequence(outputs_bw, seqlens, batch_axis=0, seq_axis=1)

        return outputs_bw, outputs_fw



class DBLSTM(object):

    def __init__(self, X, T, seqlens):
        self._initialize_layers([32, 32, 32])

        self.X = X
        self.T = T
        self.Tflat = tf.cast(tf.argmax(T, 2), tf.int32)
        self.seqlens = seqlens

        self.prediction
        self.optimize
        self.error

        # self.layers = []
        # for i, sz in enumerate(layers_sizes):
        #     if i == 0:
        #         self.layers.append(DataLayer(sz))
        #     else:
        #         self.layers.append(HiddenLayer(sz))

        # build graph variables & placeholders
        # Wo_shape = (2 * layers_sizes[-1], target_size)
        # bo_shape = (target_size)
        # self.Wo = tf.Variable(tf.random_normal(Wo_shape, name='Wo'))
        # self.bo = tf.Variable(tf.random_normal(bo_shape, name='bo'))

        # # pipeline control place holders
        # # This makes training slower - but code is reusable
        # X_shape = (None, None, feature_size)
        # T_shape = (None, None, target_size)
        # self.X = tf.placeholder(tf.float32, shape=X_shape, name='X')
        # self.T = tf.placeholder(tf.float32, shape=T_shape, name='T')
        # self.minibatch = tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size

    def _initialize_layers(self, layers_sizes):
        if self.layers is None:
            self.layers = []
            for i, sz in enumerate(layers_sizes):
                if i == 0:
                    self.layers.append(DataLayer(sz))
                else:
                    self.layers.append(HiddenLayer(sz))

    @lazy_property
    def prediction(self):
        with tf.variable_scope('forward_0', reuse=tf.AUTO_REUSE):
            outputs, hidden = self.layers[0].prediction(self.X, self.seqlens)

        for i, layer in enumerate(self.layers[1:]):
            with tf.variable_scope('forward_{:}'.format(i+1), reuse=tf.AUTO_REUSE):
                outputs, hidden = layer.prediction(outputs, hidden)

        with tf.variable_scope('score'):
            outputs = tf.concat((outputs, hidden), axis=2)

        # Stacking is cleaner and faster - but it's hard to use for multiple pipelines
        score = tf.scan(
            lambda a, x: tf.matmul(x, Wo),
            outputs, initializer=tf.matmul(outputs[0], Wo)) + bo
        return score

    @lazy_property
    def optimize(self):
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.prediction, self.Tflat, self.seqlens)

        # Compute the viterbi sequence and score.
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.prediction, transition_params, self.seqlens)

        cost_op = tf.reduce_mean(-log_likelihood)

        return tf.train.AdamOptimizer(learning_rate=5 * 1e-3).minimize(self.cost)

    @lazy_property
    def cost(self):
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.prediction, self.Tflat, self.seqlens)

        # Compute the viterbi sequence and score.
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.prediction, transition_params, self.seqlens)

        return tf.reduce_mean(-log_likelihood)

    @lazy_property
    def error(self):
        return error_rate2(self.prediction, self.Tflat, self.error)

def main():
    # PARAMETERS
    layers_sizes = [32, 32, 32]
    lr = 5 * 1e-3
    feature_size = 1 * 2 + 50 * (2 + 3)
    target_size = 30
    batch_size = 250
    num_epochs = 10
    input_sequence_features = ['ID', 'FORM', 'LEMMA', 'PRED_MARKER', 'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']
    target = 'T'
    # mnist = input_data.read_data_sets('./mnist/', one_hot=True)


    Wo_shape = (2 * layers_sizes[-1], target_size)
    bo_shape = (target_size)

    Wo = tf.Variable(tf.random_normal(Wo_shape, name='Wo'))
    bo = tf.Variable(tf.random_normal(bo_shape, name='bo'))

    # pipeline control place holders
    # This makes training slower - but code is reusable
    X_shape = (None, None, feature_size)
    T_shape = (None, None, target_size)
    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    seqlens = tf.placeholder(tf.int32, shape=(None,), name='seqlens')

    with tf.name_scope('pipeline'):
        inputs, targets, sequence_length, descriptors = input_fn(
            [DATASET_TRAIN_GLO50_PATH], batch_size, num_epochs,
            input_sequence_features, target)

    dblstm = DBLSTM(X, T, seqlens)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Training control variables
        step = 0
        total_loss = 0.0
        total_error = 0.0

        try:
            while not coord.should_stop():
                X_batch, Y_batch, L_batch, D_batch = session.run([inputs, targets, sequence_length, descriptors])


                _, loss, error = session.run(
                    dblstm.optimize, dblstm.cost, dblstm.error,
                    feed_dict={X: X_batch, T: Y_batch, seqlens: L_batch}
                )

                total_loss += loss
                total_error += error
                if (step + 1) % 5 == 0:
                    print('Iter={:5d}'.format(step + 1),
                          '\tavg. cost {:.6f}'.format(total_loss / 5),
                          '\tavg. error {:.6f}'.format(total_error / 5))
                    total_loss = 0.0
                    total_error = 0.0
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()