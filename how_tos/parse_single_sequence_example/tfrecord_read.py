'''
Created on Jan 28, 2018
	@author: Varela
	
	Using tensorflow's iterator

'''
import tensorflow as tf 
from tensorflow.python import debug as tf_debug

TARGET_PATH='how_tos/parse_single_sequence_example/datasets/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'

def read_and_decode(filename_queue):
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# print(serialized_example.shape)
	context_features, sequence_features= tf.parse_single_sequence_example(
		serialized_example,
		context_features={
			'length': tf.FixedLenFeature([], tf.int64)
			# 'PRED':tf.FixedLenFeature([], tf.int64)
			# 'LEMMA':tf.FixedLenFeature([], tf.int64),
			# 'M_R':tf.FixedLenFeature([], tf.int64),
			# 'target':tf.FixedLenFeature([], tf.int64)		
		}
	)

	# print(context_features['length'].shape())
	length= 	 tf.cast(context_features['length'], tf.int32)	
	# length= 	 tf.cast(context_features['length'], tf.int32)	
	# print(length)
	# predicate= tf.constant(features['PRED'], tf.int32)	
	return length

if __name__== '__main__':
	filename_queue= tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
	
	# length, predicate= read_and_decode(filename_queue)
	length= read_and_decode(filename_queue)
	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)

		for i in range(90):
			# length, predicate= session.run([length, predicate])
			l= session.run([length])
			print(l)
			# print(predicate)

		coord.request_stop()
		coord.join(threads)

