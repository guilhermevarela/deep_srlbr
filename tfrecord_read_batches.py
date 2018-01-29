'''
Created on Jan 29, 2018
	@author: Varela
	
	Using tensorflow's Coordinator/Queue
	Using batching

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload, embed_output_lazyload  

TARGET_PATH='datasets/training/pre/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'

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

	Y= tf.nn.embedding_lookup(klass_ind, targets)

	M_R= tf.expand_dims(mr, 2)

	X= tf.concat((LEMMA, PRED, M_R), 2)
	return X, Y

# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_pipeline(filenames, batch_size,  num_epochs, embeddings, klass_ind):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	length,  idx_pred, idx_lemma,  mr, targets= read_and_decode(filename_queue)	

	X, Y = process_example(length,  idx_pred, idx_lemma,  mr, targets, embeddings, klass_ind)

	return X, Y 	



if __name__== '__main__':
	word2idx, _embeddings= embed_input_lazyload()		
	klass2idx, _klass_ind= embed_output_lazyload()		

	embeddings= tf.constant(_embeddings.tolist(), shape=(len(_embeddings),50), dtype=tf.float32)
	klass_ind= tf.constant(_klass_ind.tolist(), shape=_klass_ind.shape, dtype=tf.float32)

	X, Y = input_pipeline([tfrecords_filename], 100, 1, embeddings, klass_ind)
	# filename_queue= tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
	
	# length,  idx_pred, idx_lemma,  M_R, targets= read_and_decode(filename_queue)	
		

	#LEMMA<TIME,1,EMBEDDING_SIZE> --> <TIME,EMBEDDING_SIZE>
	# LEMMA  = tf.nn.embedding_lookup(embeddings, idx_lemma)
	#PRED<EMBEDDING_SIZE,> --> <1,EMBEDDING_SIZE> 
	# PRED   = tf.nn.embedding_lookup(embeddings, idx_pred)

	# TARGET = tf.nn.embedding_lookup(klass_ind, targets)

	# M_R= tf.expand_dims(M_R, 2)

	# Xi= tf.concat((LEMMA, PRED, M_R), 2)

	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)

		for i in range(1):
		
			X, Y =session.run([X, Y])
			

			# print('length',T)
			# print('predicate',PRED.shape)
			# # print('norm predicate',NORM_PRED.shape)
			# print('lemma',LEMMA.shape)
			# # print('norm m_r',NORM_M_R.shape)
			
			# print('mr', M_R.shape)
			print('X',X.shape)
			print('Y',Y.shape)
			


		coord.request_stop()
		coord.join(threads)

