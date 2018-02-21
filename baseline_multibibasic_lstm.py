'''
Created on Jan 30, 2018
	@author: Varela
	
	compute a 2-layered bi-lstm with (almost) non linguist task
	
	# lr=5e-4	
	# HIDDEN_SIZE=[128, 64]
	# TRAINING
	# Iter=25451 avg. acc 99.77% avg. cost 0.004194
	# VALIDATION
	# Iter= 3050 avg. acc 79.68% valid. acc 60.62% avg. cost 0.471232
	# lr=5e-4	
	# HIDDEN_SIZE=[128, 64]


	updates:
		2018-02-03: patched cross entropy and accuracy according to
		https://danijar.com/variable-sequence-lengths-in-tensorflow/
		2018-02-07: validation set included, save and load 
		2018-02-08: updates on data transformation
		2018-02-20: updates on pipeline

'''
import sys
sys.path.append('datasets/')

import numpy as np 
import tensorflow as tf 

from data_tfrecords import input_fn
from data_outputs import  dir_getoutputs, mapper_get, outputs_settings_persist, outputs_predictions_persist
from utils import cross_entropy, error_rate, precision, recall 

# INPUT_PATH='datasets/inputs/00/'
# INPUT_PATH='datasets/inputs/01/'
INPUT_PATH='datasets/inputs/02/'
dataset_train= INPUT_PATH + 'train.tfrecords'
dataset_valid= INPUT_PATH + 'valid.tfrecords'

MODEL_NAME='multi_bibasic_lstm'
DATASET_VALID_SIZE= 569
DATASET_TRAIN_SIZE= 5099


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

if __name__== '__main__':	
	#BEST RUNNING PARAMS 	
	# TRAINING
	# Iter=25451 avg. acc 99.77% avg. cost 0.004194
	# lr=5e-4	
	# HIDDEN_SIZE=[128, 64]

	lr=5e-4	
	HIDDEN_SIZE=[128, 64]

	EMBEDDING_SIZE=50 
	KLASS_SIZE=22

	FEATURE_SIZE=2*EMBEDDING_SIZE+2
	
	BATCH_SIZE=250
	N_EPOCHS=300
	
	DISPLAY_STEP=50

	
	load_dir=''
	#UNCOMMENT IN TO KEEP TRAINING
	# load_dir= 'models/multi_bibasic_lstm/lr5.00e-04,hs128x64/00/exp-1449.meta'
	# experiment_dir= 'models/multi_bibasic_lstm/lr5.00e-04,hs128x64/06/'
	# experiment_dir= dir_getmodels(lr, HIDDEN_SIZE, model_name=MODEL_NAME)
	# logs_dir= 'logs/multi_bibasic_lstm/lr5.00e-04,hs128x64/06/'
	# logs_dir= dir_getlogs(lr, HIDDEN_SIZE, model_name=MODEL_NAME)	

	outputs_dir= dir_getoutputs(lr, HIDDEN_SIZE, model_name=MODEL_NAME)	

	# print('experiment_dir', experiment_dir)
	# print('logs_dir', logs_dir)
	print('outputs_dir', outputs_dir)
	outputs_settings_persist(outputs_dir, dict(globals(), **locals()))

	klass2idx, word2idx, embeddings= mapper_get('LEMMA', 'ARG_1', INPUT_PATH)
	embeddings= tf.constant(embeddings.tolist(), shape=embeddings.shape, dtype=tf.float32, name= 'embeddings')

	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1], KLASS_SIZE], name='Wo')) 
	bo = tf.Variable(tf.random_normal([KLASS_SIZE], name='bo')) 

	#Forward backward weights for bi-lstm act
	Wfb = tf.Variable(tf.random_normal([2*HIDDEN_SIZE[-1], HIDDEN_SIZE[-1]], name='Wfb')) 
	bfb = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1]], name='bfb')) 

	#pipeline control place holders
	# This makes training slower - but code is reusable
	X     =   tf.placeholder(tf.float32, shape=(None,None, FEATURE_SIZE), name='X') 
	T     =   tf.placeholder(tf.float32, shape=(None,None, KLASS_SIZE), name='T')
	minibatch    =   tf.placeholder(tf.int32, shape=(None,), name='minibatch') # mini batches size


	#Architecture
	fwd_cell = tf.nn.rnn_cell.MultiRNNCell(
		[ tf.nn.rnn_cell.BasicLSTMCell(hsz, forget_bias=1.0, state_is_tuple=True) 
			for hsz in HIDDEN_SIZE],
	)
	bwd_cell = tf.nn.rnn_cell.MultiRNNCell(
		[ tf.nn.rnn_cell.BasicLSTMCell(hsz,  forget_bias=1.0, state_is_tuple=True) 
			for hsz in HIDDEN_SIZE],
	)
	
	#output metrics
	loss_avg= tf.placeholder(tf.float32, name='loss_avg')	
	accuracy_avg= tf.placeholder(tf.float32, name='accuracy_avg')	
	accuracy_valid= tf.placeholder(tf.float32, name='accuracy_valid')	
	logits=   tf.placeholder(tf.float32, shape=(BATCH_SIZE,None, KLASS_SIZE), name='logits')
	

	with tf.name_scope('pipeline'):
		inputs, targets, sequence_length, descriptors = input_fn([dataset_train], BATCH_SIZE, N_EPOCHS, embeddings, klass_size=KLASS_SIZE)
		inputs_v, targets_v, sequence_length_v, descriptors_v = input_fn([dataset_valid], DATASET_VALID_SIZE, 1, embeddings, klass_size=KLASS_SIZE)
	
	with tf.name_scope('predict'):		
		predict_op= forward(X, minibatch)

	with tf.name_scope('xent'):
		probs=tf.nn.softmax(tf.clip_by_value(predict_op,clip_value_min=-22,clip_value_max=22))
		argmax_op=  tf.argmax(probs, 2)
		cost_op=cross_entropy(probs, T)

	with tf.name_scope('train'):
		optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	#Evaluation
	with tf.name_scope('evaluation'):
		accuracy_op = 1.0-error_rate(probs, T, minibatch)
		precision_op=precision(probs, T)
		recall_op=recall(probs, T) 
		f1_op= 2* precision_op * recall_op/(precision_op + recall_op)



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
					[optimizer_op, predict_op, cost_op, accuracy_op],
						feed_dict= { X:X_batch, T:Y_batch, minibatch:mb}
				)
				
				total_loss+=loss 
				total_acc+= acc
				
				if (step+1) % DISPLAY_STEP ==0:					
					#This will be caugth by input_fn				
					acc, Yhat_valid  = session.run(
						[accuracy_op, argmax_op],
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
							saver.save(session, outputs_dir + 'exp', global_step=step, write_meta_graph=True)
							first_save=False 
						else:
							saver.save(session, outputs_dir + 'exp', global_step=step, write_meta_graph=True)
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

