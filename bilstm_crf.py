'''
Created on Feb 23, 2018
	@author: Varela
	
	compute a 1-layered (depth 2) bi-lstm with non linguist task

CTX-P=0
Iter= 6050 avg. acc 96.01% valid. acc 74.29% avg. cost 1.990351

Number of Sentences    :         326
Number of Propositions :         553
Percentage of perfect props :   4.70

              corr.  excess  missed    prec.    rec.      F1
------------------------------------------------------------
   Overall      398    2068     866    16.14   31.49   21.34
----------
        A0      124     285     130    30.32   48.82   37.41
        A1      202    1312     288    13.34   41.22   20.16
        A2       18     179     169     9.14    9.63    9.37
        A3        1      14      15     6.67    6.25    6.45
        A4        2      14       9    12.50   18.18   14.81
    AM-ADV        4      19      19    17.39   17.39   17.39
    AM-CAU        0      16      17     0.00    0.00    0.00
    AM-DIR        0       0       1     0.00    0.00    0.00
    AM-DIS        6      17      20    26.09   23.08   24.49
    AM-EXT        0       3       5     0.00    0.00    0.00
    AM-LOC        9      65      46    12.16   16.36   13.95
    AM-MNR        0      24      28     0.00    0.00    0.00
    AM-NEG       10       5      24    66.67   29.41   40.82
    AM-PNC        0       9       9     0.00    0.00    0.00
    AM-PRD        0      15      18     0.00    0.00    0.00
    AM-TMP       22      91      68    19.47   24.44   21.67
------------------------------------------------------------
         V      457      32      96    93.46   82.64   87.72
------------------------------------------------------------

lr5.00e-04_hs512x64_ctx-p1 .:
Iter= 7200 avg. acc 99.85% valid. acc 65.60% avg. cost 0.084763

Number of Sentences    :         326
Number of Propositions :         553
Percentage of perfect props :  15.37

              corr.  excess  missed    prec.    rec.      F1
------------------------------------------------------------
   Overall      464     894     800    34.17   36.71   35.39
----------
        A0      152     150     102    50.33   59.84   54.68
        A1      208     378     282    35.49   42.45   38.66
        A2       35     117     152    23.03   18.72   20.65
        A3        1      11      15     8.33    6.25    7.14
        A4        3      10       8    23.08   27.27   25.00
    AM-ADV        3       7      20    30.00   13.04   18.18
    AM-CAU        1      25      16     3.85    5.88    4.65
    AM-DIR        0       1       1     0.00    0.00    0.00
    AM-DIS        5      14      21    26.32   19.23   22.22
    AM-EXT        1       2       4    33.33   20.00   25.00
    AM-LOC        8      58      47    12.12   14.55   13.22
    AM-MED        0       1       0     0.00    0.00    0.00
    AM-MNR        1      45      27     2.17    3.57    2.70
    AM-NEG       26       6       8    81.25   76.47   78.79
    AM-PNC        1       9       8    10.00   11.11   10.53
    AM-PRD        1       9      17    10.00    5.56    7.14
    AM-REC        0       1       0     0.00    0.00    0.00
    AM-TMP       18      50      72    26.47   20.00   22.78
------------------------------------------------------------
         V      520      19      33    96.47   94.03   95.24
------------------------------------------------------------

lr5.00e-04_hs512x64_ctx-p1 .: after embedding upgrades


'''
import sys
sys.path.append('datasets/')

import numpy as np 
import tensorflow as tf 
import argparse
import re

from data_tfrecords import input_fn, input_sz
from data_outputs import  dir_getoutputs, mapper_getiodicts, outputs_settings_persist, outputs_predictions_persist
from utils import cross_entropy, error_rate2, precision, recall


INPUT_PATH='datasets/inputs/01/'
# INPUT_PATH='datasets/inputs/00/'
dataset_train= INPUT_PATH + 'train.tfrecords'
dataset_valid= INPUT_PATH + 'valid.tfrecords'

MODEL_NAME='bilstm_crf'
DATASET_VALID_SIZE= 569
DATASET_TRAIN_SIZE= 5099

LAYER_1_NAME='glove_s50'
LAYER_2_NAME='bi-lstm'
LAYER_3_NAME='crf'

#Command line defaults 
LEARNING_RATE=5e-4	
HIDDEN_SIZE=[512, 64]
EMBEDDING_SIZE=50 
EMBEDDING_MODEL=[('glove','50')]
BATCH_SIZE=250
N_EPOCHS=500

def forward(X, sequence_length):		
	'''
		Computes forward propagation thru basic lstm cell

		args:
			X: [batch_size, max_time, feature_size] tensor (sequences shorter than max_time are zero padded)

			sequence_length:[batch_size] tensor (int) carrying the size of each sequence 

		returns:
			Y_hat: [batch_size, max_time, klass_size] 

	'''

	# batch_size=tf.size(sequence_length)	
	# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
	outputs, states= tf.nn.bidirectional_dynamic_rnn(
			cell_fw=fwd_cell, 
			cell_bw=bwd_cell, 
			inputs=X, 			
			sequence_length=sequence_length,
			dtype=tf.float32,
			time_major=False
		)

	fwd_outputs, bck_outputs = outputs

	outputs=tf.concat((fwd_outputs,bck_outputs),2)

	# Stacking is cleaner and faster - but it's hard to use for multiple pipelines
	# act = tf.matmul(tf.concat((fwd_outputs,bck_outputs),2), tf.stack([Wfb]*batch_size)) +bfb
	act =tf.scan(lambda a, x: tf.matmul(x, Wfb), outputs, initializer=tf.matmul(outputs[0],Wfb))+bfb

	# Stacking is cleaner and faster - but it's hard to use for multiple pipelines
	#Yhat=tf.matmul(act, tf.stack([Wo]*batch_size)) + bo
	Yhat=tf.scan(lambda a, x: tf.matmul(x, Wo),act, initializer=tf.matmul(act[0],Wo)) + bo

	return Yhat


# def main(hidden_size, 
# 	embedding_name, embedding_size, ctx_p, lr, batch_size, num_epochs):


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
  
	#BEST RUNNING PARAMS 	
	# TRAINING
	# Iter= 6050 avg. acc 96.01% valid. acc 74.29% avg. cost 1.990351
	# lr=5e-4	
	# HIDDEN_SIZE=[128, 64]

	# lr=5e-4	
	# HIDDEN_SIZE=[512, 64]
	# EMBEDDING_SIZE=50 

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
	LAYER_1_NAME=embeddings_id

	print(hidden_size, embeddings_name, embeddings_size, ctx_p, lr, batch_size, num_epochs)


	input_sequence_features= ['ID', 'LEMMA', 'M_R', 'PRED']  
	if ctx_p > 0:
		input_sequence_features+=['CTX_P{:+d}'.format(i) 
			for i in range(-ctx_p,ctx_p+1) if i !=0 ]

	feature_size=input_sz(input_sequence_features, embeddings_size)		
	
	
	load_dir=''
	outputs_dir= dir_getoutputs(lr, hidden_size, ctx_p=ctx_p, embeddings_id=embeddings_id, model_name=MODEL_NAME)	
	

	print('outputs_dir', outputs_dir)
	outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

	klass2idx, word2idx, embeddings= mapper_getiodicts('LEMMA', 'ARG_1', INPUT_PATH, embeddings_id=embeddings_id) # fetch embeddings here

	embeddings= tf.constant(embeddings.tolist(), shape=embeddings.shape, dtype=tf.float32, name= 'embeddings')
	klass_size=len(klass2idx)
	print('klass_size:', klass_size)

	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([hidden_size[-1], klass_size], name='Wo')) 
	bo = tf.Variable(tf.random_normal([klass_size], name='bo')) 

	#Forward backward weights for bi-lstm act
	Wfb = tf.Variable(tf.random_normal([2*hidden_size[-1], hidden_size[-1]], name='Wfb')) 
	bfb = tf.Variable(tf.random_normal([hidden_size[-1]], name='bfb')) 

	#pipeline control place holders
	# This makes training slower - but code is reusable
	X     =   tf.placeholder(tf.float32, shape=(None,None, feature_size), name='X') 
	T     =   tf.placeholder(tf.float32, shape=(None,None, klass_size), name='T')
	minibatch    =   tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size


	#Architecture
	fwd_cell = tf.nn.rnn_cell.MultiRNNCell(
		[ tf.nn.rnn_cell.BasicLSTMCell(hsz, forget_bias=1.0, state_is_tuple=True) 
			for hsz in hidden_size],
	)
	bwd_cell = tf.nn.rnn_cell.MultiRNNCell(
		[ tf.nn.rnn_cell.BasicLSTMCell(hsz,  forget_bias=1.0, state_is_tuple=True) 
			for hsz in hidden_size],
	)
	
	#output metrics
	loss_avg= tf.placeholder(tf.float32, name='loss_avg')	
	accuracy_avg= tf.placeholder(tf.float32, name='accuracy_avg')	
	accuracy_valid= tf.placeholder(tf.float32, name='accuracy_valid')	

	logits=   tf.placeholder(tf.float32, shape=(batch_size,None), name='logits')
	
	print('feature_size: ',feature_size)
	with tf.name_scope('pipeline'):
		inputs, targets, sequence_length, descriptors= input_fn(
			[dataset_train], batch_size, num_epochs, embeddings,
				 klass_size=klass_size, input_sequence_features=input_sequence_features)

		inputs_v, targets_v, sequence_length_v, descriptors_v= input_fn(
			[dataset_valid], DATASET_VALID_SIZE, 1, embeddings, 
				klass_size=klass_size, input_sequence_features=input_sequence_features)

	with tf.name_scope('predict'):		
		predict_op= forward(X, minibatch)
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

	#Initialization 
	#must happen after every variable has been defined
	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)

		
		# This first loop instanciates validation set
		try:
			while not coord.should_stop():				
				X_valid, Y_valid, mb_valid, D_valid=session.run([inputs_v, targets_v, sequence_length_v, descriptors_v])					

		except tf.errors.OutOfRangeError:
			print('Done initializing validation set')			

		finally:
			#When done, ask threads to stop
			coord.request_stop()			
			coord.join(threads)
			
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
							outputs_dir, D_valid[:,:,0], D_valid[:,:,1], Yhat_valid, mb_valid, klass2idx, 'Yhat_valid')

				step+=1
				
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()
			coord.join(threads)

	# input_sequence_features= ['ID', 'LEMMA', 'M_R', 'PRED']  + ['CTX_P-1', 'CTX_P+1']
	# FEATURE_SIZE=input_sz(input_sequence_features, EMBEDDING_SIZE)	
	

	# # BATCH_SIZE=250
	# # N_EPOCHS=500
	
	# DISPLAY_STEP=50

	
	# load_dir=''
	# outputs_dir= dir_getoutputs(lr, HIDDEN_SIZE, ctx_p=(len(input_sequence_features)-4)/2, model_name=MODEL_NAME)	

	# print('outputs_dir', outputs_dir)
	# outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

	# klass2idx, word2idx, embeddings= mapper_get('LEMMA', 'ARG_1', INPUT_PATH)

	# embeddings= tf.constant(embeddings.tolist(), shape=embeddings.shape, dtype=tf.float32, name= 'embeddings')
	# klass_size=len(klass2idx)
	# print('klass_size:', klass_size)

	# #define variables / placeholders
	# Wo = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1], klass_size], name='Wo')) 
	# bo = tf.Variable(tf.random_normal([klass_size], name='bo')) 

	# #Forward backward weights for bi-lstm act
	# Wfb = tf.Variable(tf.random_normal([2*HIDDEN_SIZE[-1], HIDDEN_SIZE[-1]], name='Wfb')) 
	# bfb = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1]], name='bfb')) 

	# #pipeline control place holders
	# # This makes training slower - but code is reusable
	# X     =   tf.placeholder(tf.float32, shape=(None,None, FEATURE_SIZE), name='X') 
	# T     =   tf.placeholder(tf.float32, shape=(None,None, klass_size), name='T')
	# minibatch    =   tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size


	# #Architecture
	# fwd_cell = tf.nn.rnn_cell.MultiRNNCell(
	# 	[ tf.nn.rnn_cell.BasicLSTMCell(hsz, forget_bias=1.0, state_is_tuple=True) 
	# 		for hsz in HIDDEN_SIZE],
	# )
	# bwd_cell = tf.nn.rnn_cell.MultiRNNCell(
	# 	[ tf.nn.rnn_cell.BasicLSTMCell(hsz,  forget_bias=1.0, state_is_tuple=True) 
	# 		for hsz in HIDDEN_SIZE],
	# )
	
	# #output metrics
	# loss_avg= tf.placeholder(tf.float32, name='loss_avg')	
	# accuracy_avg= tf.placeholder(tf.float32, name='accuracy_avg')	
	# accuracy_valid= tf.placeholder(tf.float32, name='accuracy_valid')	
	# # logits=   tf.placeholder(tf.float32, shape=(BATCH_SIZE,None, klass_size), name='logits')
	# logits=   tf.placeholder(tf.float32, shape=(BATCH_SIZE,None), name='logits')
	
	# print('feature_size: ',FEATURE_SIZE)
	# with tf.name_scope('pipeline'):
	# 	inputs, targets, sequence_length, descriptors = input_fn([dataset_train], BATCH_SIZE, N_EPOCHS, embeddings, klass_size=klass_size, input_sequence_features=input_sequence_features)
	# 	inputs_v, targets_v, sequence_length_v, descriptors_v = input_fn([dataset_valid], DATASET_VALID_SIZE, 1, embeddings, klass_size=klass_size, input_sequence_features=input_sequence_features)

	
	# with tf.name_scope('predict'):		
	# 	predict_op= forward(X, minibatch)
	# 	clip_prediction=tf.clip_by_value(predict_op,clip_value_min=-22,clip_value_max=22)
	# 	T_2d= tf.cast(tf.argmax( T,2 ), tf.int32)

	# with tf.name_scope('xent'):
	# 	# Compute the log-likelihood of the gold sequences and keep the transition
 #    # params for inference at test time.
	# 	log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
	# 		clip_prediction,T_2d, minibatch)

 #    # Compute the viterbi sequence and score.
	# 	viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
	# 		clip_prediction, transition_params, minibatch)


	# 	cost_op= tf.reduce_mean(-log_likelihood)
    

	# with tf.name_scope('train'):
	# 	optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	# #Evaluation
	# with tf.name_scope('evaluation'):
	# 	accuracy_op = 1.0-error_rate2(viterbi_sequence,T_2d, minibatch)



	# #Logs 
	# writer = tf.summary.FileWriter(outputs_dir)			
	# tf.summary.histogram('Wo', Wo)
	# tf.summary.histogram('bo', bo)
	# tf.summary.histogram('Wfb', Wfb)
	# tf.summary.histogram('bfb', bfb)
	# tf.summary.histogram('logits', logits)
	# tf.summary.scalar('loss_avg', loss_avg)
	# tf.summary.scalar('accuracy_avg', accuracy_avg)
	# tf.summary.scalar('accuracy_valid', accuracy_valid)
	# merged_summary = tf.summary.merge_all()

	# #Initialization 
	# #must happen after every variable has been defined
	# init_op = tf.group( 
	# 	tf.global_variables_initializer(),
	# 	tf.local_variables_initializer()
	# )
	# with tf.Session() as session: 
	# 	session.run(init_op) 
	# 	coord= tf.train.Coordinator()
	# 	threads= tf.train.start_queue_runners(coord=coord)

		
	# 	# This first loop instanciates validation set
	# 	try:
	# 		while not coord.should_stop():				
	# 			X_valid, Y_valid, mb_valid, D_valid=session.run([inputs_v, targets_v, sequence_length_v, descriptors_v])					

	# 	except tf.errors.OutOfRangeError:
	# 		print('Done initializing validation set')			

	# 	finally:
	# 		#When done, ask threads to stop
	# 		coord.request_stop()			
	# 		coord.join(threads)
			
	# with tf.Session() as session: 			
	# 	session.run(init_op) 
	# 	coord= tf.train.Coordinator()
	# 	threads= tf.train.start_queue_runners(coord=coord)
	# 	# Training control variables
	# 	step=0		
	# 	total_loss=0.0
	# 	total_acc=0.0		
		
		
	# 	#Persists always saving on improvement
	# 	if load_dir:
	# 		raise NotImplementedError('load_dir')
	# 		# saver = tf.train.import_meta_graph(load_dir)
	# 		# saver.restore(session,tf.train.latest_checkpoint('./'))
	# 		# graph = tf.get_default_graph()

	# 		# Exibits all variables
	# 		# session.graph.get_collection(tf.GraphKeys.VARIABLES) 
	# 	else:			
	# 		saver = tf.train.Saver(max_to_keep=1)
		

	# 	first_save=True
	# 	best_validation_rate=-1
	# 	writer.add_graph(session.graph)
	# 	try:
	# 		while not coord.should_stop():				
	# 			X_batch, Y_batch, mb = session.run(
	# 				[inputs, targets, sequence_length]
	# 			)

	# 			_, Yhat, loss, acc = session.run(
	# 				[optimizer_op, viterbi_sequence, cost_op, accuracy_op],
	# 					feed_dict= { X:X_batch, T:Y_batch, minibatch:mb}
	# 			)
				
	# 			total_loss+=loss 
	# 			total_acc+= acc				
	# 			if (step+1) % DISPLAY_STEP ==0:					
	# 				#This will be caugth by input_fn				
	# 				acc, Yhat_valid= session.run(
	# 					[accuracy_op, viterbi_sequence],
	# 					feed_dict={X:X_valid, T:Y_valid, minibatch:mb_valid}
	# 				)
					
	# 				#Broadcasts to user
	# 				print('Iter={:5d}'.format(step+1),
	# 					'avg. acc {:.2f}%'.format(100*total_acc/DISPLAY_STEP),						
	# 						'valid. acc {:.2f}%'.format(100*acc),						
	# 				 			'avg. cost {:.6f}'.format(total_loss/DISPLAY_STEP))										
	# 				total_loss=0.0 
	# 				total_acc=0.0					

	# 				#Logs the summary
	# 				s= session.run(merged_summary,
	# 					feed_dict={accuracy_avg: float(total_acc)/DISPLAY_STEP , accuracy_valid: acc, loss_avg: float(total_loss)/DISPLAY_STEP, logits:Yhat}
	# 				)
	# 				writer.add_summary(s, step)
	# 				if best_validation_rate < acc:
	# 					if first_save:
	# 						saver.save(session, outputs_dir + 'graph_params', global_step=step, write_meta_graph=True)
	# 						first_save=False 
	# 					else:
	# 						saver.save(session, outputs_dir + 'graph_params', global_step=step, write_meta_graph=True)
	# 					best_validation_rate = acc	

	# 					outputs_predictions_persist(
	# 						outputs_dir, D_valid[:,:,0], D_valid[:,:,1], Yhat_valid, mb_valid, klass2idx, 'Yhat_valid')

	# 			step+=1
				
	# 	except tf.errors.OutOfRangeError:
	# 		print('Done training -- epoch limit reached')

	# 	finally:
	# 		#When done, ask threads to stop
	# 		coord.request_stop()
	# 		coord.join(threads)

