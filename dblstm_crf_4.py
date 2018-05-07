'''
Created on Mar 02, 2018
    @author: Varela
    
    implementation of (Zhou,et Xu, 2015)

    ref:
    http://www.aclweb.org/anthology/P15-1109

    updates
    2018-03-15 Refactor major updates
    2018-04-18 version 4  input_with_embeddings_fn without processing
'''
# outputs_dir outputs/dblstm_crf_2/lr5.00e-04_hs64_ctx-p1_glove_s50/08/
# Iter= 8200 avg. acc 94.98% valid. acc 67.95% avg. cost 2.276543
# outputs_dir outputs/dblstm_crf_2/lr5.00e-04_hs32_ctx-p1_glove_s50/00/
# Iter= 5650 avg. acc 82.50% valid. acc 68.17% avg. cost 5.824657
import sys
sys.path.append('datasets/')
sys.path.append('models/')

import numpy as np 
import tensorflow as tf 
import argparse
import re

# this is not recommended but where just importing a bunch of constants
from config import *

from data_tfrecords import input_fn, tfrecords_extract_v2
from models.propbank_encoder import PropbankEncoder
from models.propbank_mappers import MapperTensor2Column, MapperT2ARG
from models.evaluator_conll import EvaluatorConll
from models.evaluator import Evaluator
from data_outputs import  dir_getoutputs, outputs_settings_persist

from utils import cross_entropy, error_rate2, precision, recall


MODEL_NAME = 'dblstm_crf_4'
LAYER_1_NAME = 'glove_s50'
LAYER_2_NAME = 'dblstm'
LAYER_3_NAME = 'crf'


# Command line defaults
LEARNING_RATE = 5e-4
HIDDEN_SIZE = [512, 64]
EMBEDDING_SIZE = 50
EMBEDDING_MODEL = [('glove', 50)]
BATCH_SIZE = 250
N_EPOCHS = 500

def get_cell(sz):
    return tf.nn.rnn_cell.BasicLSTMCell(sz, forget_bias=1.0, state_is_tuple=True)

def dblstm_1(X, seqlens, sz):
    with tf.variable_scope('fw'):
        # CREATE / REUSE FWD/BWD CELL
        cell_fw = get_cell(sz)

        outputs_fw, states = tf.nn.dynamic_rnn(
            cell=cell_fw,
            inputs=X,
            sequence_length=seqlens,
            dtype=tf.float32,
            time_major=False
        )

    with tf.variable_scope('bw'):
        cell_bw = get_cell(sz)
        inputs_bw = tf.reverse_sequence(outputs_fw, seqlens, batch_axis=0, seq_axis=1)

        outputs_bw, states= tf.nn.dynamic_rnn(
            cell=cell_bw, 
            inputs=inputs_bw,           
            sequence_length=seqlens,
            dtype=tf.float32,
            time_major=False
        )
        outputs_bw = tf.reverse_sequence(outputs_bw, seqlens, batch_axis=0, seq_axis=1)

    return outputs_bw, outputs_fw


def dblstm_n(outputs, h, seqlens, sz):


    with tf.variable_scope('fw'):
        inputs_fw = tf.concat((outputs, h), axis=2)
        # CREATE / REUSE FWD/BWD CELL
        cell_fw = get_cell(sz)

        outputs_fw, states = tf.nn.dynamic_rnn(
            cell=cell_fw,
            inputs=inputs_fw,
            sequence_length=seqlens,
            dtype=tf.float32,
            time_major=False
        )

    with tf.variable_scope('bw'):
        cell_bw=    get_cell(sz)
        inputs_bw = tf.concat((outputs_fw, outputs), axis=2)
        inputs_bw = tf.reverse_sequence(inputs_bw, seqlens, batch_axis=0, seq_axis=1)
        
        outputs_bw, states= tf.nn.dynamic_rnn(
            cell=cell_bw, 
            inputs=inputs_bw,           
            sequence_length=seqlens,
            dtype=tf.float32,
            time_major=False
        )
        outputs_bw = tf.reverse_sequence(outputs_bw, seqlens, batch_axis=0, seq_axis=1)

    return outputs_bw, outputs_fw

def forward(X, sequence_length, hidden_size):
    '''
        Computes forward propagation thru basic lstm cell

        args:
            X: [batch_size, max_time, feature_size] tensor (sequences shorter than max_time are zero padded)

            sequence_length:[batch_size] tensor (int) carrying the size of each sequence 

        returns:
            Y_hat: [batch_size, max_time, target_size] 

    '''
    outputs = X
    with tf.variable_scope('dblstm_1', reuse=tf.AUTO_REUSE):
        outputs, h = dblstm_1(outputs, sequence_length, hidden_size[0])

    for i, sz in enumerate(hidden_size[1:]):
        with tf.variable_scope('dblstm_{:}'.format(i+2), reuse=tf.AUTO_REUSE):
            outputs, h = dblstm_n(outputs, h, sequence_length, sz)

    with tf.variable_scope('score'):   
        outputs = tf.concat((outputs, h), axis=2)        

        # Stacking is cleaner and faster - but it's hard to use for multiple pipelines
        #Yhat=tf.matmul(act, tf.stack([Wo]*batch_size)) + bo
        score=tf.scan(lambda a, x: tf.matmul(x, Wo),
                outputs, initializer=tf.matmul(outputs[0],Wo)) + bo
    return score



if __name__== '__main__':   

    def check_embeddings(value):        
        if not(re.search(r'^[a-z0-9]*\_s\d+$', value)):
            raise argparse.ArgumentTypeError("{:} is an invalid embeddings".format(value))
        else:
            embeddings_name, embeddings_size = value.split('_s')
        return embeddings_name, int(embeddings_size)

    #Parse descriptors 
    parser = argparse.ArgumentParser(
    description='''Script used for customizing inputs for the bi-LSTM model and using CRF.''')

    parser.add_argument('depth',  type=int, nargs='+',default=HIDDEN_SIZE,
                    help='''Set of integers corresponding the layer sizes on MultiRNNCell\n''')

    parser.add_argument('--embeddings', dest='embeddings_model', type=check_embeddings, nargs=1, default=EMBEDDING_MODEL,
                    help='''embedding model name and size in format 
                    <embedding_name>_s<embedding_size>. Examples: glove_s50, wang2vec_s100\n''')

    parser.add_argument('--ctx_p', dest='ctx_p', type=int, nargs=1, default=1, choices=[0,1,2,3],
                    help='''Size of sliding window around predicate\n''')

    parser.add_argument('--lr', dest='lr', type=float, nargs=1, default=LEARNING_RATE,
                    help='''Learning rate of the model\n''')

    parser.add_argument('--batch_size', dest='batch_size', type=int, nargs=1, default=BATCH_SIZE,
                    help='''Group up to batch size propositions during training.\n''')

    parser.add_argument('--epochs', dest='epochs', type=int, nargs=1, default=N_EPOCHS,
                    help='''Number of times to repeat training set during training.\n''')

    args = parser.parse_args()
    hidden_size = args.depth
    embeddings_name, embeddings_size = args.embeddings_model[0]

    model_alias = 'wrd' if embeddings_name == 'word2vec' else embeddings_name[:3]




    # evaluate embedding model
    ctx_p = args.ctx_p[0] if isinstance(args.ctx_p, list) else args.ctx_p
    lr = args.lr[0] if isinstance(args.lr, list) else args.lr
    batch_size = args.batch_size[0] if isinstance(args.batch_size, list) else args.batch_size
    num_epochs = args.epochs[0] if isinstance(args.epochs, list) else args.epochs
    embeddings_id = '{:}_s{:}'.format(embeddings_name, embeddings_size) # update LAYER_1_NAME
    DISPLAY_STEP = 50
    target = 'T'

    PROP_DIR = './datasets/binaries/'

    PROP_PATH = '{:}deep_{:}{:}.pickle'.format(PROP_DIR, model_alias, embeddings_size)
    propbank_encoder = PropbankEncoder.recover(PROP_PATH)
    tensor2column = MapperTensor2Column(propbank_encoder)
    t2arg = MapperT2ARG(propbank_encoder)
    print('propbank_encoder columns {:}'.format(propbank_encoder.columns))

    # Updata settings
    LAYER_1_NAME = embeddings_id
    HIDDEN_SIZE = hidden_size
    BATCH_SIZE = batch_size
    EMBEDDING_SIZE = embeddings_size
    INPUT_TRAIN_PATH = '{:}dbtrain_{:}{:}.tfrecords'.format(INPUT_DIR, model_alias, embeddings_size)
    INPUT_VALID_PATH = '{:}dbvalid_{:}{:}.tfrecords'.format(INPUT_DIR, model_alias, embeddings_size)

    print(hidden_size, embeddings_name, embeddings_size, ctx_p, lr, batch_size, num_epochs)


    input_sequence_features = ['ID', 'FORM', 'LEMMA', 'GPOS', 'PRED_MARKER', 'FORM_CTX_P+0',
        'LEMMA_CTX_P-1','LEMMA_CTX_P+0','LEMMA_CTX_P+1',
        'GPOS_CTX_P-1','GPOS_CTX_P+0','GPOS_CTX_P+1']

    if ctx_p > 0:
        input_sequence_features+=['FORM_CTX_P{:+d}'.format(i) 
            for i in range(-ctx_p,ctx_p+1) if i !=0 ]



    columns_dimensions = propbank_encoder.columns_dimensions('EMB')
    feature_size = sum([
        columns_dimensions[col]
        for col in input_sequence_features
    ])
    target_size = columns_dimensions[target]
    target2idx = propbank_encoder.lex2idx[target]

    load_dir = ''
    outputs_dir = dir_getoutputs(lr, hidden_size, ctx_p=ctx_p, embeddings_id=embeddings_id, model_name=MODEL_NAME)

    print('outputs_dir', outputs_dir)
    outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

    print('{:}_size:{:}'.format(target, columns_dimensions[target]))

    # calculator_train = Evaluator(propbank.column('train', 'T', True))
    calculator_train = Evaluator(propbank_encoder.column('train', 'T', 'CAT'))
    evaluator_train = EvaluatorConll(
        'train',
        propbank_encoder.column('train', 'S', 'CAT'),
        propbank_encoder.column('train', 'P', 'CAT'),
        propbank_encoder.column('train', 'PRED', 'CAT'),
        propbank_encoder.column('train', 'ARG', 'CAT'),
        outputs_dir
    )

    X_train, T_train, mb_train, D_train = tfrecords_extract_v2(
        'train', 
        input_sequence_features, 
        target
    )
    
    calculator_valid=Evaluator(propbank_encoder.column('valid', 'T', 'CAT'))
    evaluator_valid= EvaluatorConll(
        'valid', 
        propbank_encoder.column('valid', 'S', 'CAT'),
        propbank_encoder.column('valid', 'P', 'CAT'),
        propbank_encoder.column('valid', 'PRED', 'CAT'),
        propbank_encoder.column('valid', 'ARG', 'CAT'),
        outputs_dir
    )
    
    X_valid, T_valid, mb_valid, D_valid=tfrecords_extract_v2(
        'valid', 
        input_sequence_features, 
        target
    )


    # define variables / placeholders
    Wo = tf.Variable(tf.random_normal([2*hidden_size[-1], target_size], name='Wo')) 
    bo = tf.Variable(tf.random_normal([target_size], name='bo')) 

    # pipeline control place holders
    # This makes training slower - but code is reusable
    X = tf.placeholder(tf.float32, shape=(None,None, feature_size), name='X') 
    T = tf.placeholder(tf.float32, shape=(None,None, target_size), name='T')
    minibatch = tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size


    print('feature_size: ',feature_size)
    with tf.name_scope('pipeline'):
        inputs, targets, sequence_length, descriptors = input_fn(
            [DATASET_TRAIN_V2_PATH.replace('_pt_v2','_glo50')], batch_size, num_epochs,
            input_sequence_features, target)

    with tf.name_scope('predict'):
        predict_op = forward(X, minibatch, hidden_size)
        # clip_prediction=tf.clip_by_value(predict_op,clip_value_min=-22,clip_value_max=22)
        Tflat = tf.cast(tf.argmax(T, 2), tf.int32)

    with tf.name_scope('xent'):
        # Compute the log-likelihood of the gold sequences and xkeep the transition
        # params for inference at test time.
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            predict_op, Tflat, minibatch)

        # Compute the viterbi sequence and score.
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            predict_op, transition_params, minibatch)

        cost_op = tf.reduce_mean(-log_likelihood)

    with tf.name_scope('train'):
        optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

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

        first_save = True
        best_validation_rate = -1

        try:
            while not coord.should_stop():
                X_batch, Y_batch, mb, D_batch = session.run(
                    [inputs, targets, sequence_length, descriptors]
                )


                _, Yhat, loss = session.run(
                    [optimizer_op, viterbi_sequence, cost_op],
                    feed_dict= { X: X_batch, T: Y_batch, minibatch: mb}
                )

                total_loss += loss
                if (step + 1) % DISPLAY_STEP == 0:
                    # This will be caugth by input_with_embeddings_fn
                    Yhat = session.run(
                        viterbi_sequence,
                        feed_dict={ X: X_train,
                                    T: T_train,
                                    minibatch: mb_train
                                   }
                    )
                
                    index = D_train[:, :, 0].astype(np.int32)

                    # predictions_d = propbank_encoder.tensor2column(
                    #     index, Yhat, mb_train, 'T')
                    # acc_train = calculator_train.accuracy(predictions_d)

                    # predictions_d = propbank_encoder.t2arg(predictions_d)
                    predictions_d = tensor2column.define(index, Yhat, mb_train, 'T').map()
                    acc_train = calculator_train.accuracy(predictions_d)
                    predictions_d = t2arg.define(predictions_d, 'CAT').map()
                    
                    evaluator_train.evaluate( predictions_d, True)

                    Yhat = session.run(
                        viterbi_sequence,
                        feed_dict={X: X_valid, T: T_valid, minibatch: mb_valid}
                    )

                    index = D_valid[:, :, 0].astype(np.int32)
                    # predictions_d = propbank_encoder.tensor2column(
                    #     index, Yhat, mb_valid, 'T')
                    # acc_valid = calculator_valid.accuracy(predictions_d)
                    # predictions_d = propbank_encoder.t2arg(predictions_d)
                    # evaluator_valid.evaluate(predictions_d, False)
                    predictions_d = tensor2column.define(index, Yhat, mb_valid, 'T').map()
                    acc_valid = calculator_valid.accuracy(predictions_d)
                    predictions_d = t2arg.define(predictions_d, 'CAT').map()
                    evaluator_valid.evaluate(predictions_d, False)


                    print('Iter={:5d}'.format(step + 1),
                          'train-f1 {:.2f}%'.format(evaluator_train.f1),
                          'avg acc {:.2f}%'.format(100 * acc_train),
                          'valid-f1 {:.2f}%'.format(evaluator_valid.f1),
                          'valid acc {:.2f}%'.format(100 * acc_valid),
                          'avg. cost {:.6f}'.format(total_loss / DISPLAY_STEP))
                    total_loss = 0.0
                    total_acc = 0.0

                    if best_validation_rate < evaluator_valid.f1:
                        best_validation_rate = evaluator_valid.f1
                        evaluator_valid.evaluate(predictions_d, True)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)
