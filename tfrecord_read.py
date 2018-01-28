'''
Created on Jan 28, 2018
	@author: Varela
	
	Using tensorflow's iterator

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload 

TARGET_PATH='datasets/training/pre/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'

def read_and_decode(filename_queue):
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# 
	context_features, sequence_features= tf.parse_single_sequence_example(
		serialized_example,
		context_features={
			'length': tf.FixedLenFeature([], tf.int64),
			'PRED':tf.FixedLenFeature([], tf.int64)			
		},
		sequence_features={
			'LEMMA': tf.VarLenFeature(tf.int64),
			'M_R':tf.VarLenFeature(tf.int64),
			'target':tf.VarLenFeature(tf.int64)		
		}
	)

	# print(context_features['length'].shape())
	length= 	 	 tf.cast(context_features['length'], tf.int32)	
	predicate= 	 tf.cast(context_features['PRED'], tf.int32)	


	#tf.VarLenFeatures are mapped to dense arrays
	lemma= 	  tf.sparse_tensor_to_dense(sequence_features['LEMMA'])	
	mr= 	 		tf.sparse_tensor_to_dense(sequence_features['M_R'])	
	target= 	tf.sparse_tensor_to_dense(sequence_features['target'])	


	# print(length)
	# predicate= tf.constant(features['PRED'], tf.int32)	
	return length, predicate, lemma, mr, target 

if __name__== '__main__':
		word2idx, _embeddings= embed_input_lazyload()		
	embeddings= tf.constant(_embeddings.tolist(), shape=(len(_embeddings),50), dtype=tf.float32)
	
	filename_queue= tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
	
	length, idx_pred, idx_lemma, M_R, target= read_and_decode(filename_queue)	
		

	LEMMA = tf.nn.embedding_lookup(embeddings, idx_lemma)
	PRED  = tf.nn.embedding_lookup(embeddings, idx_pred)

	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)

		for i in range(1):
			# length, predicate= session.run([length, predicate])
			l, PRED, LEMMA, M_R, t = session.run([length, PRED, LEMMA, M_R, target])
			

			print('length',l)
			print('predicate',PRED.shape)
			print('lemma',LEMMA.shape)
			print('mr', M_R.shape)
			print('target',t)


		coord.request_stop()
		coord.join(threads)

