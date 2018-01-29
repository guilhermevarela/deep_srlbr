'''
Created on Jan 28, 2018
	@author: Varela
	
	Using tensorflow's iterator

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload, embed_output_lazyload  

TARGET_PATH='datasets/training/pre/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'

def read_and_decode(filename_queue):
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# 
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
	mr= 	 		tf.sparse_tensor_to_dense(sequence_features['M_R'])	
	#target<TIME,> of integers
	# targets_sparse= sequence_features['targets']
	targets= 	tf.sparse_tensor_to_dense(sequence_features['targets'])	


	return length, predicate, lemma, mr, targets 

if __name__== '__main__':
	word2idx, _embeddings= embed_input_lazyload()		
	klass2idx, _klass_ind= embed_output_lazyload()		

	embeddings= tf.constant(_embeddings.tolist(), shape=(len(_embeddings),50), dtype=tf.float32)
	klass_ind= tf.constant(_klass_ind.tolist(), shape=_klass_ind.shape, dtype=tf.float32)

	filename_queue= tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
	
	length,  idx_pred, idx_lemma,  M_R, targets= read_and_decode(filename_queue)	
		

	#LEMMA<TIME,1,EMBEDDING_SIZE> --> <TIME,EMBEDDING_SIZE>
	LEMMA  = tf.nn.embedding_lookup(embeddings, idx_lemma)
	#PRED<EMBEDDING_SIZE,> --> <1,EMBEDDING_SIZE> 
	PRED   = tf.nn.embedding_lookup(embeddings, idx_pred)

	TARGET = tf.nn.embedding_lookup(klass_ind, targets)

	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)

		for i in range(1):
		
			T, PRED, LEMMA,  M_R, TARGET =session.run([length, PRED, LEMMA,  M_R, TARGET])
			

			print('length',T)
			print('predicate',PRED.shape)
			# print('norm predicate',NORM_PRED.shape)
			print('lemma',LEMMA.shape)
			# print('norm m_r',NORM_M_R.shape)
			
			print('mr', M_R.shape)
			print('target',TARGET.shape)


		coord.request_stop()
		coord.join(threads)

