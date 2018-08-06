'''
    Created on Jun 06, 2018

    @author: Varela

    Runs training and prediction in order to estimate the parameters

'''
import tensorflow as tf
import config
import numpy as np
from .utils import get_dims, get_index, get_binary
from datasets import get_valid, get_test, input_fn, get_train
from models.propbank_encoder import PropbankEncoder
from models.evaluator_conll import EvaluatorConll
from models.dblstm import DBLSTM

# FEATURE_LABELS = ['ID', 'FORM', 'LEMMA', 'MARKER', 'GPOS',
#                   'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1',
#                   'GPOS_CTX_P-1', 'GPOS_CTX_P+0', 'GPOS_CTX_P+1']

FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']

TARGET_LABEL = 'T'

HIDDEN_LAYERS = [16] * 4


def estimate_kfold(input_labels=FEATURE_LABELS, target_label=TARGET_LABEL,
                   hidden_layers=HIDDEN_LAYERS, embeddings='wan50',
                   epochs=100, lr=5 * 1e-3, fold=25, **kwargs):
    '''Runs estimate DBLSTM parameters using a kfold cross validation

    Estimates DBLSTM using Stochastic Gradient Descent. The 
    development set in constructed by merging training
    and validation set, and each example is evaluated
    once every `fold` epochs. Evalutions for the model are
    carried out using 2005 CoNLL Shared Task Scripts and
    the best proposition forecast and scores are saved on 
    a dedicated folder.

    Keyword Arguments:
        input_labels {list<str>}-- Columns which will be 
            converted into features (default: {FEATURE_LABELS})
        target_label {str} -- Column which will be
            used as target (default: {`T`})
        hidden_layers {list<int>} -- sets the number and
            sizes for hidden layers (default: {`[16, 16, 16, 16]`})
        embeddings {str} -- There are three available models
            `glo50`,`wrd50`, `wan50` (default: {'wan50'})
        epochs {int} -- Number of iterations (default: {100})
        lr {float} -- The learning rate (default: {5 * 1e-3})
        kfold {int} -- The number of partitions from
            on each iteration (default: {250})
        **kwargs {dict<str,<key>>} -- unlisted arguments
    '''

    propbank_encoder = PropbankEncoder.recover(get_binary('deep', embeddings))
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    datasets_list = [get_binary('train', embeddings)]
    datasets_list.append(get_binary('valid', embeddings))
    dataset_size = config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE

    X_test, T_test, L_test, D_test = get_test(input_labels, target_label)
    feature_size = get_dims(input_labels, dims_dict)
    target_size = dims_dict[target_label]

    batch_size = int(dataset_size / fold)
    print(batch_size, target_label, target_size, feature_size)

    evaluator = EvaluatorConll(propbank_encoder.db, propbank_encoder.idx2lex)
    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'target_size': target_size
    }
    # BUILDING the execution graph
    X_shape = (None, None, feature_size)
    T_shape = (None, None, target_size)
    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    seqlens = tf.placeholder(tf.int32, shape=(None,), name='seqlens')

    with tf.name_scope('pipeline'):
        inputs, targets, sequence_length, descriptors = input_fn(
            datasets_list, batch_size, epochs,
            input_labels, target_label, shuffle=True)

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
        step = 0
        i = 0
        total_loss = 0.0
        total_error = 0.0
        best_validation_rate = -1
        try:
            while not coord.should_stop():
                X_batch, Y_batch, L_batch, D_batch = session.run([inputs, targets, sequence_length, descriptors])

                if  step % fold == i:
                    X_valid, Y_valid, L_valid, D_valid = X_batch, Y_batch, L_batch, D_batch

                else:
                    loss, _, Yish, error = session.run(
                        [dblstm.cost, dblstm.optimize, dblstm.prediction, dblstm.error],
                        feed_dict={X: X_batch, T: Y_batch, seqlens: L_batch}
                    )

                    total_loss += loss
                    total_error += error

                if (step + 1) % fold == 0:
                    Yish = session.run(
                        dblstm.prediction,
                        feed_dict={X: X_valid, T: Y_valid, seqlens: L_valid}
                    )

                    index = D_valid[:, :, 0].astype(np.int32)

                    evaluator.evaluate_tensor('valid', index, Yish, L_valid, target_label, params)

                    print('Iter={:5d}'.format(step + 1),
                          '\tavg. cost {:.6f}'.format(total_loss / 24),
                          '\tavg. error {:.6f}'.format(total_error / 24),
                          '\tf1-train {:.6f}'.format(evaluator.f1))

                    total_loss = 0.0
                    total_error = 0.0

                    if best_validation_rate < evaluator.f1:
                        best_validation_rate = evaluator.f1

                    if evaluator.f1 > 95:
                        Yish = session.run(
                            dblstm.prediction,
                            feed_dict={X: X_test, T: Y_test, seqlens: L_test}
                        )

                        index = D_test[:, :, 0].astype(np.int32)

                        evaluator.evaluate_tensor('test', index, Yish, L_test, target_label, params)
                step += 1
                i = int(step / fold) % fold

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)


def estimate(input_labels=FEATURE_LABELS, target_label=TARGET_LABEL,
             hidden_layers=HIDDEN_LAYERS, embeddings='wan50',
             epochs=100, lr=5 * 1e-3, batch_size=250, **kwargs):
    '''Runs estimate DBLSTM parameters using a training set and a fixed validation set

    Estimates DBLSTM using Stochastic Gradient Descent. Both training 
    set and validation sets are fixed. Evalutions for the model are
    carried out using 2005 CoNLL Shared Task Scripts and the best 
    proposition forecast and scores are saved on a dedicated folder.

    Keyword Arguments:
        input_labels {list<str>}-- Columns which will be 
            converted into features (default: {FEATURE_LABELS})
        target_label {str} -- Column which will be
            used as target (default: {`T`})
        hidden_layers {list<int>} -- sets the number and
            sizes for hidden layers (default: {`[16, 16, 16, 16]`})
        embeddings {str} -- There are three available models
            `glo50`,`wrd50`, `wan50` (default: {'wan50'})
        epochs {int} -- Number of iterations (default: {100})
        lr {float} -- The learning rate (default: {5 * 1e-3})
        batch_size {int} -- The number of examples consumed 
            on each iteration (default: {250})
        **kwargs {dict<str,<key>>} -- unlisted arguments
    '''

    propbank_encoder = PropbankEncoder.recover(get_binary('deep', embeddings))
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    datasets_list = [get_binary('train', embeddings)]

    labels_list = input_labels + [target_label]

    X_train, T_train, L_train, I_train = get_train(input_labels, target_label, embeddings)
    X_valid, T_valid, L_valid, I_valid = get_valid(input_labels, target_label, embeddings)
    feature_size = get_dims(input_labels, dims_dict)
    target_size = dims_dict[target_label]

    # print(batch_size, target_label, target_size, feature_size)    
    def train_eval(Y):
        index = I_train[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor('train', index, Y, L_train, target_label, params)

        return evaluator.f1

    def valid_eval(Y, prefix='valid'):
        index = I_valid[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor(prefix, index, Y, L_valid, target_label, params)

        return evaluator.f1

    evaluator = EvaluatorConll(propbank_encoder.db, propbank_encoder.idx2lex)
    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'target_size': target_size
    }
    # BUILDING the execution graph
    X_shape = (None, None, feature_size)
    T_shape = (None, None, target_size)
    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    seqlens = tf.placeholder(tf.int32, shape=(None,), name='seqlens')

    with tf.name_scope('pipeline'):
        inputs, targets, sequence_length, descriptors = input_fn(
            datasets_list, batch_size, epochs,
            input_labels, target_label, shuffle=True)

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
                        f1_valid = valid_eval(Y_valid, 'best-valid')

                step += 1


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)