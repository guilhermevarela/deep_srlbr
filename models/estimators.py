'''
    Created on Jun 06, 2018

    @author: Varela

    Runs training and prediction in order to estimate the parameters

'''
import os


import numpy as np
import tensorflow as tf

import config

from utils.info import get_db_bounds, get_binary
from utils.snapshots import snapshot_hparam_string, snapshot_persist, snapshot_recover
# from utils.ml import f1_score

from models.propbank_encoder import PropbankEncoder
from models.conll_evaluator import ConllEvaluator
from models.labelers import Labeler, DualLabeler
from models.streamers import TfStreamer




FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']

TARGET_LABEL = 'T'

HIDDEN_LAYERS = [16] * 4


def estimate_kfold(input_labels=FEATURE_LABELS, target_labels=TARGET_LABEL,
                   hidden_layers=HIDDEN_LAYERS, embeddings_model='wan50',
                   version='1.0', epochs=100, lr=5 * 1e-3, fold=25, ctx_p=1,
                   ckpt_dir=None,  ru='BasicLSTM', chunks=False, r_depth=-1, **kwargs):
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
        target_labels {str} -- Column which will be
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
        target_dir = snapshot_hparam_string(embeddings_model=embeddings_model,
                                            target_labels=target_labels,
                                            is_batch=False, ctx_p=ctx_p,
                                            learning_rate=lr, version=version,
                                            hidden_layers=hidden_layers)

        target_dir = 'outputs{:}'.format(target_dir)
        target_dir = snapshot_persist(
            target_dir,
            input_labels=input_labels, lr=lr,
            hidden_layers=hidden_layers, ctx_p=ctx_p,
            target_labels=target_labels, kfold=25,
            embeddings_trainable=False,
            embeddings_model=embeddings_model, ru=ru,
            epochs=epochs, chunks=chunks, r_depth=r_depth,
            version=version)
    else:
        target_dir = ckpt_dir

    save_path = '{:}model.ckpt'.format(target_dir)
    propbank_encoder = PropbankEncoder.recover(get_binary('deep', embeddings_model))

    datasets_list = [get_binary('train', embeddings_model)]
    datasets_list.append(get_binary('valid', embeddings_model))

    _, dataset_size = get_db_bounds('valid', version=version)
    

    cnf_dict = config.get_config(embeddings_model)

    X_test, T_test, L_test, D_test = TfStreamer.get_test(
        input_labels, target_labels, version=version,
        embeddings_model=embeddings_model
    )
    feature_size = sum([cnf_dict[lbl]['size'] for lbl in input_labels])
    targets_size = max([cnf_dict[lbl]['size'] for lbl in target_labels])

    batch_size = int(dataset_size / fold)
    print(batch_size, target_labels, targets_size, feature_size)

    evaluator = ConllEvaluator(propbank_encoder, target_dir=target_dir)
    targets_size = [cnf_dict[lbl]['size'] for lbl in target_labels]
    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'targets_size': targets_size,
        'ru': ru
    }
    # BUILDING the execution graph
    X_shape = get_xshape(input_labels, cnf_dict)
    T_shape = get_tshape(target_labels, cnf_dict)
    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    L = tf.placeholder(tf.int32, shape=(None,), name='L')

    streamer = TfStreamer(datasets_list, batch_size, epochs,
                          input_labels, target_labels, shuffle=True)

    if len(target_labels) == 1:
        deep_srl = Labeler(X, T, L, **params)
    elif len(target_labels) == 2:
        deep_srl = DualLabeler(X, T, L, r_depth, **params)
    else:
        err = 'len(target_labels) <= 2 got {:}'.format(len(target_labels))
        raise ValueError(err)

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
                X_batch, Y_batch, L_batch, D_batch = session.run(streamer.stream)


                if  step % fold == i:
                    X_valid, Y_valid, L_valid, D_valid = X_batch, Y_batch, L_batch, D_batch

                else:
                    loss, _, Yish, error = session.run(
                        [deep_srl.cost, deep_srl.label, deep_srl.predict, deep_srl.error],
                        feed_dict={X: X_batch, T: Y_batch, L: L_batch}
                    )

                    total_loss += loss
                    total_error += error

                if (step + 1) % fold == 0:
                    Yish = session.run(
                        deep_srl.predict,
                        feed_dict={X: X_valid, T: Y_valid, L: L_valid}
                    )

                    # index = D_valid[:, :, 0].astype(np.int32)
                    if len(target_labels) == 2:
                        Yish = Yish[1]
                        labels = target_labels[1:]
                    else:
                        labels = target_labels
                    evaluator.evaluate_npyarray('valid', D_valid, Yish, L_valid, target_labels, params)

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
                            feed_dict={X: X_test, T: Y_test, L: L_test}
                        )

                        index = D_test[:, :, 0].astype(np.int32)
                        if len(target_labels) == 2:
                            Yish = Yish[1]
                            labels = target_labels[1:]
                        else:
                            labels = target_labels

                        evaluator. evaluate_npyarray('test', index, Yish, L_test, labels, params)
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


def estimate(input_labels=FEATURE_LABELS, target_labels=TARGET_LABEL,
             hidden_layers=HIDDEN_LAYERS, embeddings_model='wan50',
             embeddings_trainable=False, epochs=100, lr=5 * 1e-3,
             batch_size=250, ctx_p=1, version='1.0', ckpt_dir=None,
             ru='BasicLSTM', chunks=False, r_depth=-1, **kwargs):
    '''Runs estimate DBLSTM parameters using a training set and a fixed validation set

    Estimates DBLSTM using Stochastic Gradient Descent. Both training 
    set and validation sets are fixed. Evalutions for the model are
    carried out using 2005 CoNLL Shared Task Scripts and the best 
    proposition forecast and scores are saved on a dedicated folder.

    Keyword Arguments:
        input_labels {list<str>}-- Columns which will be 
            converted into features (default: {FEATURE_LABELS})
        target_labels {str} -- Column which will be
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
        target_dir = snapshot_hparam_string(embeddings_model=embeddings_model,
                                            target_labels=target_labels,
                                            learning_rate=lr, is_batch=True,
                                            hidden_layers=hidden_layers,
                                            version=version, ctx_p=ctx_p)

        target_dir = 'outputs{:}'.format(target_dir)

        target_dir = snapshot_persist(
            target_dir, input_labels=input_labels,
            target_labels=target_labels, epochs=epochs,
            embeddings_model=embeddings_model, ru=ru,
            embeddings_trainable=embeddings_trainable,
            hidden_layers=hidden_layers, ctx_p=ctx_p,
            lr=lr, batch_size=batch_size, r_depth=r_depth,
            chunks=chunks, version=version)
    else:
        target_dir = ckpt_dir

    save_path = '{:}model.ckpt'.format(target_dir)
    propbank_path = get_binary('deep', embeddings_model, version=version)
    propbank_encoder = PropbankEncoder.recover(propbank_path)

    dataset_path = get_binary('train', embeddings_model, version=version)
    datasets_list = [dataset_path]

    X_train, T_train, L_train, I_train = TfStreamer.get_train(
        input_labels, target_labels, version=version,
        embeddings_model=embeddings_model
    )

    X_valid, T_valid, L_valid, I_valid = TfStreamer.get_valid(
        input_labels, target_labels, version=version,
        embeddings_model=embeddings_model
    )

    cnf_dict = config.get_config(embeddings_model)
    targets_size = [cnf_dict[lbl]['size'] for lbl in target_labels]
    params = {
        'learning_rate': lr,
        'hidden_size': hidden_layers,
        'targets_size': targets_size,
        'ru': ru
    }

    evaluator = ConllEvaluator(propbank_encoder, target_dir=target_dir)

    def train_eval(Y):
        # index = I_train[:, :, 0].astype(np.int32)
        if len(target_labels) == 2:
            Y = Y[1]
            labels = target_labels[1:]
        else:
            labels = target_labels
        evaluator. evaluate_npyarray('train', I_train, Y, L_train, labels, params, script_version='04')

        return evaluator.f1

    def valid_eval(Y, prefix='valid'):
        # index = I_valid[:, :, 0].astype(np.int32)
        if len(target_labels) == 2:
            Y = Y[1]
            labels = target_labels[1:]
        else:
            labels = target_labels
        evaluator. evaluate_npyarray(prefix, I_valid, Y, L_valid, labels, params, script_version='04')

        return evaluator.f1


    # BUILDING the execution graph
    X_shape = get_xshape(input_labels, cnf_dict)
    T_shape = get_tshape(target_labels, cnf_dict)
    print('trainable embeddings?', embeddings_trainable)
    print('X_shape', X_shape)
    print('T_shape', T_shape)

    X = tf.placeholder(tf.float32, shape=X_shape, name='X')
    T = tf.placeholder(tf.float32, shape=T_shape, name='T')
    L = tf.placeholder(tf.int32, shape=(None,), name='L')

    streamer = TfStreamer(datasets_list, batch_size, epochs,
                          input_labels, target_labels, shuffle=True)

    if len(target_labels) == 1:
        deep_srl = Labeler(X, T, L, **params)
    elif len(target_labels) == 2:
        deep_srl = DualLabeler(X, T, L, r_depth, **params)
    else:
        err = 'len(target_labels) <= 2 got {:}'.format(len(target_labels))
        raise ValueError(err)

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
                X_batch, T_batch, L_batch, I_batch = session.run(streamer.stream)
                # import code; code.interact(local=dict(globals(), **locals()))

                loss, _, Yish, error = session.run(
                    [deep_srl.cost, deep_srl.label, deep_srl.predict, deep_srl.error],
                    feed_dict={X: X_batch, T: T_batch, L: L_batch}
                )

                total_loss += loss
                total_error += error
                if (step) % 25 == 0:

                    Y_train = session.run(
                        deep_srl.predict,
                        feed_dict={X: X_train, T: T_train, L: L_train}
                    )                    
                    f1_train = train_eval(Y_train)


                    Y_valid = session.run(
                        deep_srl.predict,
                        feed_dict={X: X_valid, T: T_valid, L: L_valid}
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


def get_xshape(input_labels, cnf_dict):
    # axis 0 --> examples
    # axis 1 --> max time
    # axis 2 --> feature size
    xshape = [None, None]
    feature_sz = sum([cnf_dict[lbl]['size'] for lbl in input_labels])
    xshape.append(feature_sz)
    return xshape


def get_tshape(output_labels, cnf_dict):
    # axis 0 --> examples
    # axis 1 --> max time
    base_shape = [None, None]
    k = len(output_labels)
    if k == 1:
        # axis 2 --> target size
        m = cnf_dict[output_labels[0]]['size']
        tshape = base_shape + [m]
    elif k == 2:
        # axis 2 --> max target size
        # axis 3 --> number of targets
        m = max([cnf_dict[lbl]['size'] for lbl in output_labels])
        tshape = base_shape + [m, k]
    else:
        err = 'len(target_labels) <= 2 got {:}'.format(k)
        raise ValueError(err)
    return tshape
