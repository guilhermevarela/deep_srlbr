'''
    Created on Jun 06, 2018

    @author: Varela

    Runs optimization models
'''
import tensorflow as tf
import config
import np
from utils import get_dims, get_index, get_binary
from datasets import get_valid, get_test, input_fn
from propbank_encoder import PropbankEncoder
from evaluator import EvaluatorConll2

FEATURE_LABELS = ['ID', 'FORM', 'LEMMA', 'PRED_MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1',
                  'GPOS_CTX_P-1', 'GPOS_CTX_P+0', 'GPOS_CTX_P+1']

TARGET_LABEL = 'T'

HIDDEN_LAYERS = [16] * 4


def optimize_kfold(DeepLSTM,
                   feature_labels_list=FEATURE_LABELS,
                   target_label=TARGET_LABEL, hidden_layers=HIDDEN_LAYERS,
                   embeddings='wan50', epochs=100, lr=5 * 1e-3, fold=25,
                   *kwargs):
    propbank_encoder = PropbankEncoder.recover(get_binary('deep', embeddings))
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    datasets_list = [get_binary('train', embeddings), get_binary('valid', embeddings)]
    dataset_size =  config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE
    labels_list = feature_labels_list + [target_label]

    index_column = get_index(labels_list, dims_dict, 'INDEX')
    X_test, T_test, L_test, D_test = get_test(feature_labels_list, target_label)
    feature_size = get_dims(feature_labels_list, dims_dict)
    target_size = dims_dict[target_label]

    batch_size = int(dataset_size / fold)
    print(batch_size, target_label, target_size, index_column, feature_size)

    evaluator = EvaluatorConll2(propbank_encoder.db, propbank_encoder.idx2lex)
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
            feature_labels_list, target_label, shuffle=True)

    dblstm = DeepLSTM(X, T, seqlens, **params)

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

                    index = D_valid[:, :, index_column].astype(np.int32)

                    evaluator.evaluate_tensor('valid', index, Yish, L_valid, TARGET, params)
                    

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

                        index = D_test[:, :, index_column].astype(np.int32)

                        evaluator.evaluate_tensor('test', index, Yish, L_test, TARGET, params)
                step += 1
                i = int(step / fold) % fold

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)


def optimize(DeepLSTM,
             feature_labels_list=FEATURE_LABELS,
             target_label=TARGET_LABEL, hidden_layers=HIDDEN_LAYERS,
             embeddings='wan50', epochs=100, lr=5 * 1e-3, batch_size=250,
             *kwargs):

    propbank_encoder = PropbankEncoder.recover(get_binary('deep', embeddings))
    dims_dict = propbank_encoder.columns_dimensions('EMB')
    datasets_list = [get_binary('train', embeddings)]

    labels_list = feature_labels_list + [target_label]

    index_column = get_index(labels_list, dims_dict, 'INDEX')
    X_valid, T_valid, L_valid, D_valid = get_valid(feature_labels_list, target_label)
    feature_size = get_dims(feature_labels_list, dims_dict)
    target_size = dims_dict[target_label]

    print(batch_size, target_label, target_size, index_column, feature_size)

    evaluator = EvaluatorConll2(propbank_encoder.db, propbank_encoder.idx2lex)
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
            feature_labels_list, target_label, shuffle=True)

    dblstm = DeepLSTM(X, T, seqlens, **params)

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
        best_validation_rate = -1
        try:
            while not coord.should_stop():
                X_batch, T_batch, L_batch, D_batch = session.run([inputs, targets, sequence_length, descriptors])

                if (step + 1) % 25 == 0:
                    Yish = session.run(
                        dblstm.prediction,
                        feed_dict={X: X_batch, T: T_batch, seqlens: L_batch}
                    )

                    index = D_batch[:, :, index_column].astype(np.int32)

                    evaluator.evaluate_tensor('valid', index, Yish, L_batch, target_label, params)

                    print('Iter={:5d}'.format(step + 1),
                          '\tavg. cost {:.6f}'.format(total_loss / 25),
                          '\tavg. error {:.6f}'.format(total_error / 25),
                          '\tf1-train {:.6f}'.format(evaluator.f1))
                    total_loss = 0.0
                    total_error = 0.0

                    if best_validation_rate < evaluator.f1:
                        best_validation_rate = evaluator.f1

                    if evaluator.f1 > 95:
                        Yish = session.run(
                            dblstm.prediction,
                            feed_dict={X: X_valid, T: T_valid, seqlens: L_valid}
                        )
                else:
                    loss, _, Yish, error = session.run(
                        [dblstm.cost, dblstm.optimize, dblstm.prediction, dblstm.error],
                        feed_dict={X: X_batch, T: T_batch, seqlens: L_batch}
                    )

                    total_loss += loss
                    total_error += error
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)