'''
Created on Jan 30, 2018
	@author: Varela
	
	Using tensorflow's Coordinator/Queue
	Using batching

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload, embed_output_lazyload  

TARGET_PATH='datasets/training/pre/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'
EMBEDDING_SIZE=50 

def read_and_decode(filename_queue):
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

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

	# INT<1>
	length= 	 	 tf.cast(context_features['T'], tf.int32)	
	

	# INT<1>
	predicate= 	 tf.sparse_tensor_to_dense(sequence_features['PRED'])	
	#lemma<TIME,>

	lemma= 	  tf.sparse_tensor_to_dense(sequence_features['LEMMA'])	
	#mr<TIME,> of zeros and ones	
	mr= 	 		tf.cast( tf.sparse_tensor_to_dense(sequence_features['M_R']), dtype=tf.float32)	
	#target<TIME,> of integers
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


def forward(X, Wo, bo):		
	basic_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0)

	outputs, states= tf.nn.dynamic_rnn(
			cell=basic_cell, 
			inputs=X, 			
			dtype=tf.float32
		)

	return tf.matmul(outputs[-1], Wo) + bo

if __name__== '__main__':
	hidden_size=128
	klass_size=60
	lr=1e-4
	x_size=2*EMBEDDING_SIZE+1
	batch_size=200
	word2idx, _embeddings= embed_input_lazyload()		
	klass2idx, _klass_ind= embed_output_lazyload()		

	embeddings= tf.constant(_embeddings.tolist(), shape=(len(_embeddings),50), dtype=tf.float32)
	klass_ind= tf.constant(_klass_ind.tolist(), shape=_klass_ind.shape, dtype=tf.float32)

	inputs, targets, length_batch = input_pipeline([tfrecords_filename], batch_size, 1, embeddings, klass_ind)
	
	#define variables / placeholders
	X = tf.placeholder(np.float32, shape=(None,None,x_size), name='X')
	sequence_length = tf.placeholder(shape=(batch_size,), dtype=np.int32, name='sequence_length')

	Wo = tf.Variable(tf.random_normal([hidden_size, klass_size], name='Wo')) 
	bo = tf.Variable(tf.random_normal([klass_size], name='bo')) 
	

	predict_op= forward(X, Wo, bo)
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

		for i in range(2):
			Xbatch, Ybatch, length_batch =session.run([inputs, targets, length_batch])

	
			print('X',Xbatch.shape)
			print('Y',Ybatch.shape)
			print('S',length_batch.shape)

			Yhat, accuracy= session.run(
				[predict_op, accuracy_op],
				feed_dict={X: Xbatch}
			)
			
			print('Yhat',Yhat.shape)
			print('accuracy',Yhat.accuracy)
			

			


		coord.request_stop()
		coord.join(threads)

