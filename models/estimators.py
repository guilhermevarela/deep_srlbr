'''
    Created on Jun 06, 2018

    @author: Varela

    Runs training and prediction in order to estimate the parameters

'''
import os


import numpy as np
import tensorflow as tf

import config

from utils.info import get_dims, get_binary
from utils.snapshots import snapshot_hparam_string, snapshot_persist, snapshot_recover
from datasets import get_valid, get_test, input_fn, get_train
from models.propbank_encoder import PropbankEncoder
from models.evaluator_conll import EvaluatorConll
from models.optimizers import Optmizer




FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']

TARGET_LABEL = 'T'

HIDDEN_LAYERS = [16] * 4


def estimate_kfold(input_labels=FEATURE_LABELS, target_label=TARGET_LABEL,
                   hidden_layers=HIDDEN_LAYERS, embeddings='wan50',
                   version='1.0', epochs=100, lr=5 * 1e-3, fold=25, ctx_p=1,
                   ckpt_dir=None,  ru='BasicLSTM', chunks=False, **kwargs):
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
            on each iteration (default: 250)
        ckpt_dir {str} -- ckpt_dir is the  (default: None)
        chunks {bool} -- If true use gold standard chunks  (default: False)
        **kwargs {dict<str,<key>>} -- unlisted arguments
    '''
    if ckpt_dir is None:
        target_dir = snapshot_hparam_string(embeddings=embeddings,
                                            target_label=target_label,
                                            is_batch=False, ctx_p=ctx_p,
                                            learning_rate=lr, version=version,
                                            hidden_layers=hidden_layers)

        target_dir = 'outputs{:}'.format(target_dir)
        target_dir = snapshot_persist(target_dir,
                                      input_labels=input_labels, lr=lr,
                                      hidden_layers=hidden_layers, ctx_p=ctx_p,
                                      target_label=target_label, kfold=25,
                                      embeddings=embeddings, ru=ru,
                                      epochs=epochs, chunks=chunks,
                                      version=version)
    else:
        target_dir = ckpt_dir

    save_path = '{:}model.ckpt'.format(target_dir)
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

    evaluator = EvaluatorConll(propbank_encoder.db,
                               propbank_encoder.idx2lex, target_dir=target_dir)
    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'target_size': target_size,
        'ru': ru
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
            input_labels, target_label, shuffle=True,
            dimensions_dict=dims_dict)

    deep_srl = Optmizer(X, T, seqlens, **params)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        # Training control variables
        if ckpt_dir:
            # restores previously trained parameters
            saver.restore(session, save_path)
            conll_path = '{:}valid.conll'.format(ckpt_dir)
            evaluator.evaluate_fromconllfile(conll_path)
            best_validation_rate = evaluator.f1  # prevents files to be loaded
        else:
            best_validation_rate = -1
        step = 0
        i = 0
        total_loss = 0.0
        total_error = 0.0
        eps = 100
        try:
            while not (coord.should_stop() or eps < 1e-3):
                X_batch, Y_batch, L_batch, D_batch = session.run([inputs, targets, sequence_length, descriptors])


                if  step % fold == i:
                    X_valid, Y_valid, L_valid, D_valid = X_batch, Y_batch, L_batch, D_batch

                else:
                    loss, _, Yish, error = session.run(
                        [deep_srl.cost, deep_srl.optimize, deep_srl.predict, deep_srl.error],
                        feed_dict={X: X_batch, T: Y_batch, seqlens: L_batch}
                    )

                    total_loss += loss
                    total_error += error

                if (step + 1) % fold == 0:
                    Yish = session.run(
                        deep_srl.predict,
                        feed_dict={X: X_valid, T: Y_valid, seqlens: L_valid}
                    )

                    index = D_valid[:, :, 0].astype(np.int32)

                    evaluator.evaluate_tensor('valid', index, Yish, L_valid, target_label, params)

                    print('Iter={:5d}'.format(step + 1),
                          '\tavg. cost {:.6f}'.format(total_loss / 24),
                          '\tavg. error {:.6f}'.format(total_error / 24),
                          '\tf1-train {:.6f}'.format(evaluator.f1))

                    eps = float(total_error) / 24
                    total_loss = 0.0
                    total_error = 0.0

                    if best_validation_rate < evaluator.f1:
                        best_validation_rate = evaluator.f1
                        saver.save(session, save_path)

                    if evaluator.f1 > 95:
                        Yish = session.run(
                            deep_srl.predict,
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


def estimate_recover(ckpt_dir):
    '''Loads pre computed models and runs

    Runs estimate model under directory cktp_dir 
    from last state.

    Arguments:
        ckpt_dir {str} -- directory with checkpoint files

    Raises:
        ValueError -- ckpt_dir must be a directory
        os.FileNotFoundError -- ckpt_dir must contain 
                    valid checkpoint files and params.json
    '''

    if not os.path.isdir(ckpt_dir):
        raise ValueError('ckpt_dir `{:}` is not a directory.'.format(ckpt_dir))
    else:
        params_path = '{:}params.json'.format(ckpt_dir)
        if not os.path.isfile(params_path):
            err_ = 'params.json not in `{:}`'.format(params_path)
            raise ValueError(err_)
        else:
            ckpt_path = '{:}checkpoint'.format(ckpt_dir)
            if not os.path.isfile(ckpt_path):
                err_ = 'checkpoint not in `{:}`'.format(ckpt_path)
                raise ValueError(err_)
            else:
                params_dict = snapshot_recover(ckpt_dir)
                params_dict['ckpt_dir'] = ckpt_dir
                if 'kfold' in params_dict:
                    estimate_kfold(**params_dict)
                elif 'batch_size' in params_dict:
                    estimate(**params_dict)


def estimate(input_labels=FEATURE_LABELS, target_label=TARGET_LABEL,
             hidden_layers=HIDDEN_LAYERS, embeddings='wan50',
             epochs=100, lr=5 * 1e-3, batch_size=250, ctx_p=1,
             version='1.0', ckpt_dir=None, ru='BasicLSTM', chunks=False,
             **kwargs):
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
        ckpt_dir {str} -- ckpt_dir is the  (default: None)
        chunks {bool} -- If true use gold standard chunks  (default: False)
        **kwargs {dict<str,<key>>} -- unlisted arguments
    '''

    if ckpt_dir is None:
        target_dir = snapshot_hparam_string(embeddings=embeddings,
                                            target_label=target_label,
                                            learning_rate=lr, is_batch=True,
                                            hidden_layers=hidden_layers,
                                            version=version, ctx_p=ctx_p)

        target_dir = 'outputs{:}'.format(target_dir)

        target_dir = snapshot_persist(target_dir, input_labels=input_labels,
                                      target_label=target_label, ctx_p=ctx_p,
                                      embeddings=embeddings, epochs=epochs,
                                      hidden_layers=hidden_layers, ru=ru,
                                      lr=lr, batch_size=batch_size,
                                      chunks=chunks, version=version)
    else:
        target_dir = ckpt_dir

    save_path = '{:}model.ckpt'.format(target_dir)
    propbank_path = get_binary('deep', embeddings, version=version)
    propbank_encoder = PropbankEncoder.recover(propbank_path)
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    dataset_path = get_binary('train', embeddings, version=version)
    datasets_list = [dataset_path]

    # Get the train and valid set in memory for evaluation
    X_train, T_train, L_train, I_train = get_train(
        input_labels, target_label, embeddings=embeddings,
        dimensions_dict=dims_dict, version=version
    )


    X_valid, T_valid, L_valid, I_valid = get_valid(
        input_labels, target_label, embeddings=embeddings,
        dimensions_dict=dims_dict, version=version
    )
    feature_size = get_dims(input_labels, dims_dict)
    target_size = dims_dict[target_label]

    evaluator = EvaluatorConll(propbank_encoder.db, propbank_encoder.idx2lex, target_dir=target_dir)

    def train_eval(Y):
        index = I_train[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor('train', index, Y, L_train, target_label, params, script_version='04')

        return evaluator.f1

    def valid_eval(Y, prefix='valid'):
        index = I_valid[:, :, 0].astype(np.int32)
        evaluator.evaluate_tensor(prefix, index, Y, L_valid, target_label, params, script_version='04')

        return evaluator.f1

    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'target_size': target_size,
        'ru': ru
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
            input_labels, target_label, shuffle=True,
            dimensions_dict=dims_dict)

    # deep_srl = DBLSTM(X, T, seqlens, **params)
    deep_srl = Optmizer(X, T, seqlens, **params)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        if ckpt_dir:
            saver.restore(session, save_path)
            conll_path = '{:}best-valid.conll'.format(ckpt_dir)
            evaluator.evaluate_fromconllfile(conll_path)
            best_validation_rate = evaluator.f1
        else:
            best_validation_rate = -1

        # Training control variables
        step = 1
        total_loss = 0.0
        total_error = 0.0

        eps = 100
        try:
            while not (coord.should_stop() or eps < 1e-3):
                X_batch, T_batch, L_batch, I_batch = session.run([
                    inputs, targets, sequence_length, descriptors
                ])


                loss, _, Yish, error = session.run(
                    [deep_srl.cost, deep_srl.optimize, deep_srl.predict, deep_srl.error],
                    feed_dict={X: X_batch, T: T_batch, seqlens: L_batch}
                )

                total_loss += loss
                total_error += error
                if (step) % 25 == 0:

                    Y_train = session.run(
                        deep_srl.predict,
                        feed_dict={X: X_train, T: T_train, seqlens: L_train}
                    )
                    f1_train = train_eval(Y_train)

                    Y_valid = session.run(
                        deep_srl.predict,
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
                    eps = float(total_error) / 25
                    total_loss = 0.0
                    total_error = 0.0

                    if f1_valid and best_validation_rate < f1_valid:

                        best_validation_rate = f1_valid
                        f1_valid = valid_eval(Y_valid, 'best-valid')
                        saver.save(session, save_path)

                step += 1


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)