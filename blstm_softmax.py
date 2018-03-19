'''
Created on Mar 02, 2018
	@author: Varela
	
	implementation of (Zhou,et Xu, 2015)
	
	ref:
	http://www.aclweb.org/anthology/P15-1109

	updates
	2018-03-15 Refactor major updates
'''
# outputs_dir outputs/blstm_crf_2/lr5.00e-04_hs64_ctx-p1_glove_s50/08/
# Iter= 8200 avg. acc 94.98% valid. acc 67.95% avg. cost 2.276543
# outputs_dir outputs/blstm_crf_2/lr5.00e-04_hs32_ctx-p1_glove_s50/00/
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

from data_tfrecords import input_fn, tfrecords_extract
from models.propbank import Propbank
from models.evaluator_conll import EvaluatorConll
from models.evaluator import Evaluator
from data_outputs import  dir_getoutputs, outputs_settings_persist, outputs_predictions_persist
from data_propbankbr import propbankbr_t2arg
from utils import cross_entropy, error_rate2, precision, recall


MODEL_NAME='blstm_softmax'
LAYER_1_NAME='glove_s50'
LAYER_2_NAME='blstm'
LAYER_3_NAME='softmax'

#Command line defaults 
LEARNING_RATE=5e-4	
HIDDEN_SIZE=[512, 64]
EMBEDDING_SIZE=50 
EMBEDDING_MODEL=[('glove',50)]
BATCH_SIZE=250
N_EPOCHS=500


def get_cell(sz):
	return tf.nn.rnn_cell.BasicLSTMCell(sz,  forget_bias=1.0, state_is_tuple=True)

def blstm_layer(X, sequence_length, sz):
	'''
		refs:
			https://www.tensorflow.org/programmers_guide/variables
	'''
	# CREATE / REUSE LSTM CELL
	cell = get_cell(sz)

	# CREATE / REUSE FWD/BWD CELL
	cell_fw= 	get_cell(sz)
	cell_bw= 	get_cell(sz)

	outputs, states= tf.nn.bidirectional_dynamic_rnn(
		cell_fw=cell_fw, 
		cell_bw=cell_bw, 
		inputs=X, 			
		sequence_length=sequence_length,
		dtype=tf.float32,
		time_major=False
	)
	
	return tf.concat(outputs,2)

def forward(X, sequence_length, hidden_size):		
	'''
		Computes forward propagation thru basic lstm cell

		args:
			X: [batch_size, max_time, feature_size] tensor (sequences shorter than max_time are zero padded)

			sequence_length:[batch_size] tensor (int) carrying the size of each sequence 

		returns:
			Y_hat: [batch_size, max_time, target_size] 

	'''
	outputs=X
	for i, sz in enumerate(hidden_size):
		with tf.variable_scope('blstm_{:}'.format(i+1)):
			outputs= blstm_layer(outputs, sequence_length, sz)

	with tf.variable_scope('activation'):	
		# Stacking is cleaner and faster - but it's hard to use for multiple pipelines
		# act = tf.matmul(tf.concat((fwd_outputs,bck_outputs),2), tf.stack([Wfb]*batch_size)) +bfb		
		act =tf.scan(lambda a, x: tf.matmul(x, Wfb), 
				outputs, initializer=tf.matmul(outputs[0],Wfb))+bfb

		# Stacking is cleaner and faster - but it's hard to use for multiple pipelines
		#Yhat=tf.matmul(act, tf.stack([Wo]*batch_size)) + bo
		Yhat=tf.scan(lambda a, x: tf.matmul(x, Wo),
				act, initializer=tf.matmul(act[0],Wo)) + bo


	return Yhat



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
	hidden_size= args.depth 
	embeddings_name, embeddings_size= args.embeddings_model[0]
	




	# evaluate embedding model
	ctx_p= args.ctx_p[0] if isinstance(args.ctx_p, list) else args.ctx_p
	lr= args.lr[0] if isinstance(args.lr, list) else args.lr
	batch_size= args.batch_size[0] if isinstance(args.batch_size, list) else args.batch_size
	num_epochs= args.epochs[0] if isinstance(args.epochs, list) else args.epochs
	embeddings_id='{:}_s{:}'.format(embeddings_name, embeddings_size) # update LAYER_1_NAME
	DISPLAY_STEP=50
	target= 'T'
	
	PROP_DIR = './datasets/binaries/'
	PROP_PATH = '{:}{:}'.format(PROP_DIR, 'db_pt_LEMMA_{:}.pickle'.format(embeddings_id))
	propbank = Propbank.recover(PROP_PATH)

	# Updata settings
	LAYER_1_NAME=embeddings_id
	HIDDEN_SIZE=hidden_size
	BATCH_SIZE=batch_size
	EMBEDDING_SIZE=embeddings_size

	print(hidden_size, embeddings_name, embeddings_size, ctx_p, lr, batch_size, num_epochs)


	input_sequence_features= ['ID', 'LEMMA', 'M_R', 'PRED_1', 'GPOS']  
	if ctx_p > 0:
		input_sequence_features+=['CTX_P{:+d}'.format(i) 
			for i in range(-ctx_p,ctx_p+1) if i !=0 ]
	
	
	feature_size=propbank.size(input_sequence_features)
	hotencode2sz= {feat: propbank.size(feat)
			for feat, feat_type in META.items() if feat_type == 'hot'}		
	
	target_size=hotencode2sz[target]		
	target2idx= propbank.onehot[target]
	
	load_dir=''
	outputs_dir= dir_getoutputs(lr, hidden_size, ctx_p=ctx_p, embeddings_id=embeddings_id, model_name=MODEL_NAME)	
	

	print('outputs_dir', outputs_dir)
	outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

	print('{:}_size:{:}'.format(target, hotencode2sz[target]))

	calculator_train=Evaluator(propbank.feature('train', 'T', True))
	evaluator_train= EvaluatorConll(
		'train', 
		propbank.feature('train', 'S', True),
		propbank.feature('train', 'P', True),
		propbank.feature('train', 'PRED', True),
		propbank.feature('train', 'ARG',  True),
		outputs_dir
	)
	
	X_train, T_train, mb_train, D_train=tfrecords_extract(
		'train', 
		propbank.embeddings, 
		hotencode2sz, 
		input_sequence_features, 
		target
	)

	calculator_valid=Evaluator(propbank.feature('valid', 'T', True))	
	evaluator_valid= EvaluatorConll(
		'valid', 
		propbank.feature('valid', 'S', True),
		propbank.feature('valid', 'P', True),
		propbank.feature('valid', 'PRED', True),
		propbank.feature('valid', 'ARG', True),
		outputs_dir		
	)

	X_valid, T_valid, mb_valid, D_valid=tfrecords_extract(
		'valid', 
		propbank.embeddings, 
		hotencode2sz, 
		input_sequence_features, 
		target
	)


	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([hidden_size[-1], target_size], name='Wo')) 
	bo = tf.Variable(tf.random_normal([target_size], name='bo')) 

	#Forward backward weights for bi-lstm act
	Wfb = tf.Variable(tf.random_normal([2*hidden_size[-1], hidden_size[-1]], name='Wfb')) 
	# Wfb = tf.Variable(tf.random_normal([2*hidden_size[-1], hidden_size[-1]], name='Wfb')) 
	bfb = tf.Variable(tf.random_normal([hidden_size[-1]], name='bfb')) 

	#pipeline control place holders
	# This makes training slower - but code is reusable
	X     =   tf.placeholder(tf.float32, shape=(None,None, feature_size), name='X') 
	T     =   tf.placeholder(tf.float32, shape=(None,None, target_size), name='T')
	minibatch    =   tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size


	print('feature_size: ',feature_size)
	with tf.name_scope('pipeline'):
		inputs, targets, sequence_length, descriptors= input_fn(
			[DATASET_TRAIN_PATH], batch_size, num_epochs, 
			propbank.embeddings, hotencode2sz, 
			input_sequence_features, target)

	with tf.name_scope('predict'):		
		predict_op= forward(X, minibatch, hidden_size)

	with tf.name_scope('xent'):
		# Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
		probs=tf.nn.softmax(tf.clip_by_value(predict_op,clip_value_min=-22,clip_value_max=22))
		argmax_op=  tf.argmax(probs, 2)
		cost_op=cross_entropy(probs, T)
    

	with tf.name_scope('train'):
		optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	# Evaluation THIS ISN'T WORKING
	with tf.name_scope('evaluation'):
		accuracy_op = 1.0-error_rate(probs, T, minibatch)
		precision_op=precision(probs, T)
		recall_op=recall(probs, T) 
		f1_op= 2* precision_op * recall_op/(precision_op + recall_op)

	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
			
	with tf.Session() as session: 			
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)
		# Training control variables
		step=0		
		total_loss=0.0

		first_save=True
		best_validation_rate=-1

		try:
			while not coord.should_stop():				
				X_batch, Y_batch, mb, D_batch = session.run(
					[inputs, targets, sequence_length, descriptors]
				)
				
				_, Yhat, loss = session.run(
					[optimizer_op, viterbi_sequence, cost_op],
						feed_dict= { X:X_batch, T:Y_batch, minibatch:mb}
				)
				
				total_loss+=loss 
				if (step+1) % DISPLAY_STEP ==0:					
					#This will be caugth by input_fn				
					Yhat= session.run(
						viterbi_sequence,
						feed_dict={X:X_train, T:T_train, minibatch:mb_train}
					)					
					
					index= D_train[:,:,0]
					predictions_d= propbank.tensor2column( index, Yhat, mb_train, 'T')					
					acc_train= calculator_train.accuracy(predictions_d)
					predictions_d= propbank.t2arg(predictions_d)
					evaluator_train.evaluate( predictions_d, True )

					Yhat= session.run(
						viterbi_sequence,
						feed_dict={X:X_valid, T:T_valid, minibatch:mb_valid}
					)

					index= D_valid[:,:,0]
					predictions_d= propbank.tensor2column( index, Yhat, mb_valid, 'T')					
					acc_valid= calculator_valid.accuracy(predictions_d)
					predictions_d= propbank.t2arg(predictions_d)
					evaluator_valid.evaluate( predictions_d, False )

					print('Iter={:5d}'.format(step+1),
						'train-f1 {:.2f}%'.format(evaluator_train.f1),						
							'avg acc {:.2f}%'.format(100*acc_train),						
								'valid-f1 {:.2f}%'.format(evaluator_valid.f1),														
									'valid acc {:.2f}%'.format(100*acc_valid),						
					 					'avg. cost {:.6f}'.format(total_loss/DISPLAY_STEP))										
					total_loss=0.0 
					total_acc=0.0					

					if best_validation_rate < evaluator_valid.f1:
						best_validation_rate = evaluator_valid.f1	
						evaluator_valid.evaluate( predictions_d, True )

				step+=1
				
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()
			coord.join(threads)