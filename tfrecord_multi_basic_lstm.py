'''
Created on Jan 30, 2018
	@author: Varela
	
	Using tensorflow's Coordinator/Queue
	Using batching

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload, embed_output_lazyload  

TARGET_PATH='datasets/inputs/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'


def read_and_decode(filename_queue):
	'''
		Decodes a serialized .tfrecords containing sequences
		args
			filename_queue: [n_files] tensor containing file names which are added to queue


	'''
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# a serialized sequence example contains:
	# *context_features.: which are hold constant along the whole sequence
	#   	ex.: sequence_length
	# *sequence_features.: features that change over sequence 
	context_features, sequence_features= tf.parse_single_sequence_example(
		serialized_example,
		context_features={
			'T': tf.FixedLenFeature([], tf.int64)			
		},
		sequence_features={
			'PRED':tf.VarLenFeature(tf.int64),			
			'LEMMA': tf.VarLenFeature(tf.int64),
			'M_R':tf.VarLenFeature(tf.int64),
			'targets':tf.VarLenFeature(tf.int64)		
		}
	)

	# Returning the features length
	length= 	 	 tf.cast(context_features['T'], tf.int32)	
	
	# Predicates are constant for a sequence according to ZHOU but 
	# in order to prevent over fitting 
	predicate= 	 tf.sparse_tensor_to_dense(sequence_features['PRED'])	
	
	# lemma.: [T,]  tensor is the tokenized word expect to be different at every 	
	lemma= 	  tf.sparse_tensor_to_dense(sequence_features['LEMMA'])	
	# mr is an indicator variable which is zero is predicate has not been seen
	mr= 	 		tf.cast( tf.sparse_tensor_to_dense(sequence_features['M_R']), dtype=tf.float32)	
	
	# targets_sparse= sequence_features['targets']
	targets= 	tf.sparse_tensor_to_dense(sequence_features['targets'])	


	return length, predicate, lemma, mr, targets 


def process_example(length,  idx_pred, idx_lemma,  mr, targets, embeddings, klass_ind):
	LEMMA  = tf.nn.embedding_lookup(embeddings, idx_lemma)
	#PRED<EMBEDDING_SIZE,> --> <1,EMBEDDING_SIZE> 
	PRED   = tf.nn.embedding_lookup(embeddings, idx_pred)
	
	Y= tf.squeeze(tf.nn.embedding_lookup(klass_ind, targets),1 )

	M_R= tf.expand_dims(mr, 2)

	X= tf.squeeze( tf.concat((LEMMA, PRED, M_R), 2),1) 
	return X, Y, length

# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_pipeline(filenames, batch_size,  num_epochs, embeddings, klass_ind):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	length,  idx_pred, idx_lemma,  mr, targets= read_and_decode(filename_queue)	

	X, Y, l  = process_example(length,  idx_pred, idx_lemma,  mr, targets, embeddings, klass_ind)

	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size

	# https://www.tensorflow.org/api_docs/python/tf/train/batch
	example_batch, target_batch, length_batch=tf.train.batch(
		[X, Y, l], 
		batch_size=batch_size, 
		capacity=capacity, 
		dynamic_pad=True		
	)
	return example_batch, target_batch, length_batch


def forward(X, Wo, bo, sequence_length, hidden_sizes, batch_size):		
	'''
		Computes forward propagation thru basic lstm cell

		args:
			X: [batch_size, max_time, feature_size] tensor (sequences shorter than max_time are zero padded)

			Wo: [hidden_size[-1], klass_size] tensor prior to softmax layer, representing observable weights 

			bo: [klass_size] tensor prior to softmax layer of size 

			sequence_length:[batch_size] tensor (int) carrying the size of each sequence 

			hidden_size: [n_hidden] tensor (int) defining the hidden layer size

			batch_size: [1] tensor (int) the size of the batch

		returns:
			Yo: [batch_size, max_time, klass_size] tensor

	'''
	basic_cell = tf.nn.rnn_cell.MultiRNNCell(
		[tf.nn.rnn_cell.BasicLSTMCell(hsz, forget_bias=1.0) 
													for hsz in hidden_sizes ], state_is_tuple=True)

	# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
	outputs, states= tf.nn.dynamic_rnn(
			cell=basic_cell, 
			inputs=X, 			
			sequence_length=sequence_length,
			dtype=tf.float32,
			time_major=False
		)

	# Performs 3D tensor multiplication by stacking Wo batch_size times
	# broadcasts bias factor
	return tf.matmul(outputs, tf.stack([Wo]*batch_size)) + bo


if __name__== '__main__':	
	EMBEDDING_SIZE=50 
	KLASS_SIZE=60
	
	FEATURE_SIZE=2*EMBEDDING_SIZE+1
	lr=1e-4
	BATCH_SIZE=200	
	N_EPOCHS=20
	HIDDEN_SIZE=[256, 126]
	DISPLAY_STEP=100

	word2idx,  np_embeddings= embed_input_lazyload()		
	klass2idx, np_klassind= embed_output_lazyload()		

	embeddings= tf.constant(np_embeddings.tolist(), shape=np_embeddings.shape, dtype=tf.float32)
	klass_ind= tf.constant(np_klassind.tolist(),   shape=np_klassind.shape, dtype=tf.int32)

	inputs, targets, sequence_length = input_pipeline([tfrecords_filename], BATCH_SIZE, N_EPOCHS, embeddings, klass_ind)
	
	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1], KLASS_SIZE], name='Wo')) 
	bo = tf.Variable(tf.random_normal([KLASS_SIZE], name='bo')) 
	

	predict_op= forward(inputs, Wo, bo, sequence_length, HIDDEN_SIZE, BATCH_SIZE)
	cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_op, labels=targets))
	optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	#Evaluation
	success_count_op= tf.equal(tf.argmax(predict_op,1), tf.argmax(targets,1))
	accuracy_op = tf.reduce_mean(tf.cast(success_count_op, tf.float32))	

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
		step=0
		total_loss=0
		total_acc=0
		try:
			while not coord.should_stop():				
				_,loss, acc = session.run(
					[optimizer_op, cost_op, accuracy_op]
				)
				total_loss+=loss 
				total_acc+= acc

				if step % DISPLAY_STEP ==0:					
					print('Iter=' + str(step+1),'avg acc {:.2f}%'.format(100*total_acc/DISPLAY_STEP), 'avg cost {:.6f}'.format(total_loss/DISPLAY_STEP))
					total_loss=0
					total_acc=0
				step+=1
				
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()
			
		coord.request_stop()
		coord.join(threads)

