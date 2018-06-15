'''
    Created on May 31, 2018

    @author: Varela

    refs:
        https://danijar.com/structuring-your-tensorflow-models/
'''
import functools
import tensorflow as tf
import sys, os
sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.path.abspath('datasets'))


import config
from datasets import input_fn, get_valid, get_train
from evaluator_conll import EvaluatorConll
from propbank_encoder import PropbankEncoder
from models.utils import get_index, get_dims
import numpy as np
import yaml

INPUT_DIR = 'datasets/binaries/'
DATASET_TRAIN_GLO50_PATH = '{:}dbtrain_glo50.tfrecords'.format(INPUT_DIR)
DATASET_VALID_GLO50_PATH = '{:}dbvalid_glo50.tfrecords'.format(INPUT_DIR)
DATASET_TRAIN_WAN50_PATH = '{:}dbtrain_wan50.tfrecords'.format(INPUT_DIR)
DATASET_VALID_WAN50_PATH = '{:}dbvalid_wan50.tfrecords'.format(INPUT_DIR)
PROPBANK_GLO50_PATH = '{:}deep_glo50.pickle'.format(INPUT_DIR)
PROPBANK_WAN50_PATH = '{:}deep_wan50.pickle'.format(INPUT_DIR)




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

def get_unit(sz):
    return tf.nn.rnn_cell.BasicLSTMCell(sz, forget_bias=1.0, state_is_tuple=True)

class DBLSTM(object):

    def __init__(self, X, T, seqlens, learning_rate=5 * 1e-3, hidden_size=[32, 32], target_size=60):
        self.X = X
        self.T = T
        self.seqlens = seqlens

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.target_size = target_size

        self.propagation
        self.cost
        self.optimize
        self.prediction
        self.error



    @lazy_property
    def propagation(self):
        with tf.variable_scope('fw{:}'.format(id(self))):
            self.cell_fw = get_unit(self.hidden_size[0])

            outputs_fw, states = tf.nn.dynamic_rnn(
                cell= self.cell_fw,
                inputs=self.X,
                sequence_length=self.seqlens,
                dtype=tf.float32,
                time_major=False
            )

        with tf.variable_scope('bw{:}'.format(id(self))):
            self.cell_bw = get_unit(self.hidden_size[0])

            inputs_bw = tf.reverse_sequence(outputs_fw, self.seqlens, batch_axis=0, seq_axis=1)

            outputs_bw, states = tf.nn.dynamic_rnn(
                cell=self.cell_bw,
                inputs=inputs_bw,
                sequence_length=self.seqlens,
                dtype=tf.float32,
                time_major=False
            )
            outputs_bw = tf.reverse_sequence(outputs_bw, self.seqlens, batch_axis=0, seq_axis=1)

        h = outputs_bw
        h_1 = outputs_fw
        for i, sz in enumerate(self.hidden_size[1:]):
            n = id(self) + i + 1
            with tf.variable_scope('fw{:}'.format(n)):
                inputs_fw = tf.concat((h, h_1), axis=2)
                self.cell_fw = get_unit(sz)

                outputs_fw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_fw,
                    inputs=inputs_fw,
                    sequence_length=self.seqlens,
                    dtype=tf.float32,
                    time_major=False)

            with tf.variable_scope('bw{:}'.format(n)):
                inputs_bw = tf.concat((outputs_fw, h), axis=2)
                inputs_bw = tf.reverse_sequence(inputs_bw, self.seqlens, batch_axis=0, seq_axis=1)
                self.cell_bw = get_unit(sz)

                outputs_bw, states = tf.nn.dynamic_rnn(
                    cell=self.cell_bw,
                    inputs=inputs_bw,
                    sequence_length=self.seqlens,
                    dtype=tf.float32,
                    time_major=False)

                outputs_bw = tf.reverse_sequence(outputs_bw, self.seqlens, batch_axis=0, seq_axis=1)

            h = outputs_bw
            h_1 = outputs_fw

        with tf.variable_scope('score'):
            Wo_shape = (2 * self.hidden_size[-1], self.target_size)
            bo_shape = (self.target_size,)

            Wo = tf.Variable(tf.random_normal(Wo_shape, name='Wo'))
            bo = tf.Variable(tf.random_normal(bo_shape, name='bo'))

            outputs = tf.concat((h, h_1), axis=2)

        # Stacking is cleaner and faster - but it's hard to use for multiple pipelines
            score = tf.scan(
                lambda a, x: tf.matmul(x, Wo),
                outputs, initializer=tf.matmul(outputs[0], Wo)) + bo
        return score


    @lazy_property
    def optimize(self):
        with tf.variable_scope('optimize'):
            optimum = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        return optimum

    @lazy_property
    def cost(self):
        with tf.variable_scope('cost'):
            Tflat = tf.cast(tf.argmax(self.T, 2), tf.int32)
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.propagation, Tflat, self.seqlens)
            
        return tf.reduce_mean(-log_likelihood)

    @lazy_property
    def prediction(self):
        with tf.variable_scope('prediction'):
            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.propagation, self.transition_params, self.seqlens)

        return tf.cast(viterbi_sequence, tf.int32)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(self.prediction, tf.cast(tf.argmax(self.T, 2), tf.int32))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.T), reduction_indices=2))
        mask = tf.cast(mask, tf.float32)
        mistakes *= mask

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.seqlens, tf.float32)
        return tf.reduce_mean(mistakes)



def main():
    embeddings = 'glo50'
    propbank_encoder = PropbankEncoder.recover(PROPBANK_GLO50_PATH)
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    datasets_list = [DATASET_TRAIN_GLO50_PATH]
    input_list = ['ID', 'FORM', 'LEMMA', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1',
                  'GPOS_CTX_P-1', 'GPOS_CTX_P+0', 'GPOS_CTX_P+1']
    TARGET = 'T'
    columns_list = input_list + [TARGET]

    X_train, T_train, L_train, I_train = get_train(input_list, TARGET, embeddings)
    X_valid, T_valid, L_valid, I_valid = get_valid(input_list, TARGET, embeddings)

    FEATURE_SIZE = get_dims(input_list, dims_dict)

    BATCH_SIZE = 250
    NUM_EPOCHS = 1000
    HIDDEN_SIZE = [16] * 4
    lr = 1 * 1e-3
    TARGET_SIZE = dims_dict[TARGET]
    print(BATCH_SIZE, TARGET, TARGET_SIZE, FEATURE_SIZE)

    evaluator = EvaluatorConll(propbank_encoder.db, propbank_encoder.idx2lex)
    params = {
        'learning_rate': lr,
        'hidden_size': HIDDEN_SIZE,
        'target_size': TARGET_SIZE
    }

    def train_eval(Y):
        index = I_train[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor('train', index, Y, L_train, TARGET, params)

        return evaluator.f1

    def valid_eval(Y):
        index = I_valid[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor('valid', index, Y, L_valid, TARGET, params)

        return evaluator.f1

    # BUILDING the execution graph
    X_shape = (None, None, FEATURE_SIZE)
    T_shape = (None, None, TARGET_SIZE)
    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    seqlens = tf.placeholder(tf.int32, shape=(None,), name='seqlens')

    with tf.name_scope('pipeline'):
        inputs, targets, sequence_length, descriptors = input_fn(
            datasets_list, BATCH_SIZE, NUM_EPOCHS,
            input_list, TARGET, shuffle=True)

    dblstm = DBLSTM(X, T, seqlens, **params)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Training control variables
        step = 1
        total_loss = 0.0
        total_error = 0.0
        best_validation_rate = -1

        try:
            while not coord.should_stop():
                X_batch, T_batch, L_batch, I_batch = session.run([inputs, targets, sequence_length, descriptors])

                loss, _, Yish, error = session.run(
                    [dblstm.cost, dblstm.optimize, dblstm.prediction, dblstm.error],
                    feed_dict={X: X_batch, T: T_batch, seqlens: L_batch}
                )

                total_loss += loss
                total_error += error
                if (step) % 25 == 0:

                    Y_train = session.run(
                        dblstm.prediction,
                        feed_dict={X: X_train, T: T_train, seqlens: L_train}
                    )
                    f1_train = train_eval(Y_train)

                    Y_valid = session.run(
                        dblstm.prediction,
                        feed_dict={X: X_valid, T: T_valid, seqlens: L_valid}
                    )
                    f1_valid = valid_eval(Y_valid)

                    if f1_valid is not None and f1_train is not None:
                        print('Iter={:5d}'.format(step),
                              '\tavg. cost {:.6f}'.format(total_loss / 25),
                              '\tavg. error {:.6f}'.format(total_error / 25),
                              '\tf1-train {:.6f}'.format(f1_train),
                              '\tf1-valid {:.6f}'.format(f1_valid))
                    else:
                        print('Iter={:5d}'.format(step),
                              '\tavg. cost {:.6f}'.format(total_loss / 25),
                              '\tavg. error {:.6f}'.format(total_error / 25))
                    total_loss = 0.0
                    total_error = 0.0

                    if f1_valid and best_validation_rate < f1_valid:
                        best_validation_rate = f1_valid

                step += 1


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()
