'''
Created on Mar 02, 2018
	@author: Varela
	
	implementation of (Zhou,et Xu, 2015)
	
	ref:
	http://www.aclweb.org/anthology/P15-1109

	updates
	2018-03-15 Refactor major updates
'''
# outputs_dir outputs/dblstm_crf_2/lr5.00e-04_hs64_ctx-p1_glove_s50/08/
# Iter= 8200 avg. acc 94.98% valid. acc 67.95% avg. cost 2.276543
import sys
sys.path.append('datasets/')
sys.path.append('models/')

import numpy as np 
import tensorflow as tf 
import argparse
import re
import config as conf

from data_tfrecords_2 import input_fn, tfrecords_extract
from models.propbank import Propbank
from data_outputs import  dir_getoutputs, mapper_getiodicts, outputs_settings_persist, outputs_predictions_persist
from utils import cross_entropy, error_rate2, precision, recall

INPUT_PATH='datasets/inputs/03/'
# INPUT_PATH='datasets/inputs/01/'
# dataset_train= INPUT_PATH + 'train.tfrecords'
# dataset_valid= INPUT_PATH + 'valid.tfrecords'

dataset_train= INPUT_PATH + 'dbtrain_pt.tfrecords'
dataset_valid= INPUT_PATH + 'dbvalid_pt.tfrecords'

MODEL_NAME='dblstm_crf_2'
DATASET_VALID_SIZE= 569
DATASET_TRAIN_SIZE= 5099

LAYER_1_NAME='glove_s50'
LAYER_2_NAME='dblstm'
LAYER_3_NAME='crf'

#Command line defaults 
LEARNING_RATE=5e-4	
HIDDEN_SIZE=[512, 64]
EMBEDDING_SIZE=50 
EMBEDDING_MODEL=[('glove',50)]
BATCH_SIZE=250
N_EPOCHS=500


def get_cell(sz):
	return tf.nn.rnn_cell.BasicLSTMCell(sz,  forget_bias=1.0, state_is_tuple=True)

def dblstm_layer(X, sequence_length, sz):
	'''
		refs:
			https://www.tensorflow.org/programmers_guide/variables
	'''
	# CREATE / REUSE LSTM CELL
	cell = get_cell(sz)

	# CREATE / REUSE FWD/BWD CELL
	cell_fw= 	get_cell(sz)
	cell_bw= 	get_cell(sz)

	_outputs, states= tf.nn.dynamic_rnn(
		cell=cell, 
		inputs=X, 			
		sequence_length=sequence_length,
		dtype=tf.float32,
		time_major=False
	)
	_outputs, states= tf.nn.bidirectional_dynamic_rnn(
		cell_fw=cell_fw, 
		cell_bw=cell_bw, 
		inputs=_outputs, 			
		sequence_length=sequence_length,
		dtype=tf.float32,
		time_major=False
	)
	_, outputs_bw= tf.split(_outputs,2)

	return tf.squeeze(outputs_bw,0)

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
		with tf.variable_scope('db_lstm_{:}'.format(i+1)):
			outputs= dblstm_layer(X, sequence_length, sz)

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
	propbank = Propbank.recover(
		'db_pt_LEMMA_{:}.pickle'.format(embeddings_id))

	# Updata settings
	LAYER_1_NAME=embeddings_id
	HIDDEN_SIZE=hidden_size
	BATCH_SIZE=batch_size
	EMBEDDING_SIZE=embeddings_size

	print(hidden_size, embeddings_name, embeddings_size, ctx_p, lr, batch_size, num_epochs)


	input_sequence_features= ['ID', 'LEMMA', 'M_R', 'PRED', 'GPOS']  
	if ctx_p > 0:
		input_sequence_features+=['CTX_P{:+d}'.format(i) 
			for i in range(-ctx_p,ctx_p+1) if i !=0 ]
	
	# feature_size=input_sz(input_sequence_features, embeddings_size)		
	feature_size=propbank.size(input_sequence_features)
	hotencode2sz= {feat: propbank.size(feat)
			for feat, feat_type in conf.META.items() if feat_type == 'hot'}		
	
	target_size=hotencode2sz[target]		
	target2idx= propbank.onehot[target]
	
	load_dir=''
	outputs_dir= dir_getoutputs(lr, hidden_size, ctx_p=ctx_p, embeddings_id=embeddings_id, model_name=MODEL_NAME)	
	

	print('outputs_dir', outputs_dir)
	outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

	# target2idx, word2idx, embeddings= mapper_getiodicts('LEMMA', 'ARG_1', INPUT_PATH, embeddings_id=embeddings_id) # fetch embeddings here

	# embeddings= tf.constant(embeddings.tolist(), shape=embeddings.shape, dtype=tf.float32, name= 'embeddings')
	# target_size=len(target2idx)
	print('{:}_size:{:}'.format(target, hotencode2sz[target]))

	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([hidden_size[-1], target_size], name='Wo')) 
	bo = tf.Variable(tf.random_normal([target_size], name='bo')) 

	#Forward backward weights for bi-lstm act
	Wfb = tf.Variable(tf.random_normal([hidden_size[-1], hidden_size[-1]], name='Wfb')) 
	# Wfb = tf.Variable(tf.random_normal([2*hidden_size[-1], hidden_size[-1]], name='Wfb')) 
	bfb = tf.Variable(tf.random_normal([hidden_size[-1]], name='bfb')) 

	#pipeline control place holders
	# This makes training slower - but code is reusable
	X     =   tf.placeholder(tf.float32, shape=(None,None, feature_size), name='X') 
	T     =   tf.placeholder(tf.float32, shape=(None,None, target_size), name='T')
	minibatch    =   tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size



	
	with tf.name_scope('tensorboard'):	
		#output metrics
		loss_avg= tf.placeholder(tf.float32, name='loss_avg')	
		accuracy_avg= tf.placeholder(tf.float32, name='accuracy_avg')	
		accuracy_valid= tf.placeholder(tf.float32, name='accuracy_valid')	
		logits=   tf.placeholder(tf.float32, shape=(batch_size,None), name='logits')

	print('feature_size: ',feature_size)
	with tf.name_scope('pipeline'):
		inputs, targets, sequence_length, descriptors= input_fn(
			[dataset_train], batch_size, num_epochs, 
			propbank.embeddings, hotencode2sz, 
			input_sequence_features, target)

	with tf.name_scope('predict'):		
		predict_op= forward(X, minibatch, hidden_size)
		clip_prediction=tf.clip_by_value(predict_op,clip_value_min=-22,clip_value_max=22)
		T_2d= tf.cast(tf.argmax( T,2 ), tf.int32)

	with tf.name_scope('xent'):
		# Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
		log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
			clip_prediction,T_2d, minibatch)

    # Compute the viterbi sequence and score.
		viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
			clip_prediction, transition_params, minibatch)

		cost_op= tf.reduce_mean(-log_likelihood)
    

	with tf.name_scope('train'):
		optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	#Evaluation
	with tf.name_scope('evaluation'):
		accuracy_op = 1.0-error_rate2(viterbi_sequence,T_2d, minibatch)



	#Logs 
	writer = tf.summary.FileWriter(outputs_dir)			
	tf.summary.histogram('Wo', Wo)
	tf.summary.histogram('bo', bo)
	tf.summary.histogram('Wfb', Wfb)
	tf.summary.histogram('bfb', bfb)
	tf.summary.histogram('logits', logits)
	tf.summary.scalar('loss_avg', loss_avg)
	tf.summary.scalar('accuracy_avg', accuracy_avg)
	tf.summary.scalar('accuracy_valid', accuracy_valid)
	merged_summary = tf.summary.merge_all()

	
	X_valid, Y_valid, mb_valid, D_valid=tfrecords_extract(
		'valid', 
		propbank.embeddings, 
		hotencode2sz, 
		input_sequence_features, 
		target)


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
		total_acc=0.0		
		
		
		#Persists always saving on improvement
		if load_dir:
			raise NotImplementedError('load_dir')
			# saver = tf.train.import_meta_graph(load_dir)
			# saver.restore(session,tf.train.latest_checkpoint('./'))
			# graph = tf.get_default_graph()

			# Exibits all variables
			# session.graph.get_collection(tf.GraphKeys.VARIABLES) 
		else:			
			saver = tf.train.Saver(max_to_keep=1)
		

		first_save=True
		best_validation_rate=-1
		writer.add_graph(session.graph)
		try:
			while not coord.should_stop():				
				X_batch, Y_batch, mb = session.run(
					[inputs, targets, sequence_length]
				)
				
				
				_, Yhat, loss, acc = session.run(
					[optimizer_op, viterbi_sequence, cost_op, accuracy_op],
						feed_dict= { X:X_batch, T:Y_batch, minibatch:mb}
				)
				
				total_loss+=loss 
				total_acc+= acc				
				if (step+1) % DISPLAY_STEP ==0:					
					#This will be caugth by input_fn				
					acc, Yhat_valid= session.run(
						[accuracy_op, viterbi_sequence],
						feed_dict={X:X_valid, T:Y_valid, minibatch:mb_valid}
					)
					
					#Broadcasts to user
					print('Iter={:5d}'.format(step+1),
						'avg. acc {:.2f}%'.format(100*total_acc/DISPLAY_STEP),						
							'valid. acc {:.2f}%'.format(100*acc),						
					 			'avg. cost {:.6f}'.format(total_loss/DISPLAY_STEP))										
					total_loss=0.0 
					total_acc=0.0					

					#Logs the summary
					s= session.run(merged_summary,
						feed_dict={accuracy_avg: float(total_acc)/DISPLAY_STEP , accuracy_valid: acc, loss_avg: float(total_loss)/DISPLAY_STEP, logits:Yhat}
					)
					writer.add_summary(s, step)
					if best_validation_rate < acc:
						if first_save:
							saver.save(session, outputs_dir + 'graph_params', global_step=step, write_meta_graph=True)
							first_save=False 
						else:
							saver.save(session, outputs_dir + 'graph_params', global_step=step, write_meta_graph=True)
						best_validation_rate = acc	

						outputs_predictions_persist(
							outputs_dir, D_valid[:,:,0], D_valid[:,:,1], Yhat_valid, mb_valid, target2idx, 'Yhat_valid')

				step+=1
				
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()
			coord.join(threads)