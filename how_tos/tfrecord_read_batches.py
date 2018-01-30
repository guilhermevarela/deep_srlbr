'''
Created on Jan 29, 2018
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
	# shuffle: Boolean. If true, the strings are randomly shuffled within each epoch.
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



if __name__== '__main__':
	word2idx, _embeddings= embed_input_lazyload()		
	klass2idx, _klass_ind= embed_output_lazyload()		

	embeddings= tf.constant(_embeddings.tolist(), shape=(len(_embeddings),50), dtype=tf.float32)
	klass_ind= tf.constant(_klass_ind.tolist(), shape=_klass_ind.shape, dtype=tf.float32)

	inputs, targets, length_batch = input_pipeline([tfrecords_filename], 200, 2, embeddings, klass_ind)
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
	n_epochs= 1
	n_batches=1
	epoch_1=[]
	epoch_2=[]
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(sess=session, coord=coord)

		try:
			while not coord.should_stop():				
				X, Y, sequence_batch =session.run([inputs, targets, length_batch])
				
				#test dataset reshuffle
				if n_batches % 25==0:
					n_epochs+=1
					n_batches=0

				
				print('longest sentence', np.max(sequence_batch))
				print('total tokens', np.sum(sequence_batch))
				print('n batches', n_batches, 'n epochs', n_epochs)
				# if n_epochs==1:
				# 	epoch_1+=list(sequence_batch)
				# if n_epochs==2:
				# 	epoch_2+=list(sequence_batch)
				# n_batches+=1
		
		except tf.errors.OutOfRangeError:
			# import code; code.interact(local=dict(globals(), **locals()))			
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()


			
			# print('length',T)
			# print('predicate',PRED.shape)
			# # print('norm predicate',NORM_PRED.shape)
			# print('lemma',LEMMA.shape)
			# # print('norm m_r',NORM_M_R.shape)
			
			# print('mr', M_R.shape)
			# print('X',X.shape)
			# print('Y',Y.shape)
			# print('S',sequence_batch.shape)
			


		coord.request_stop()
		coord.join(threads)

