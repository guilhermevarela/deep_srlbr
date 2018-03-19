'''
Created on Jan 25, 2018
	@author: Varela

	Generates and reads tfrecords
		* Generates train, valid, test datasets
		* Provides input_fn that acts as a feeder


	
	2018-02-21: added input_fn
	2018-02-26: added input_sequence_features to input_fn
	2018-03-02: added input_validation and input_train
'''
import sys
ROOT_DIR = '/'.join(sys.path[0].split('/')[:-1]) #UGLY PROBLEM FIXES TO LOCAL ROOT --> import config
sys.path.append(ROOT_DIR)
sys.path.append('./models/')

import config as conf
import pandas as pd 
import numpy as np 


#Uncomment if launched from /datasets
from propbank import Propbank
# from data_propbankbr import  propbankbr_lazyload
# from data_vocabularies import vocab_lazyload_with_embeddings, vocab_lazyload, vocab_preprocess

import tensorflow as tf 

TF_SEQUENCE_FEATURES= {key:tf.VarLenFeature(tf.int64) 
	for key in conf.SEQUENCE_FEATURES
}

TF_CONTEXT_FEATURES=	{
	'L': tf.FixedLenFeature([], tf.int64)			
}




############################# tfrecords reader ############################# 
def tfrecords_extract(ds_type, embeddings, feat2size, 
										 	input_features=conf.DEFAULT_INPUT_SEQUENCE_FEATURES, 
											output_target=conf.DEFAULT_OUTPUT_SEQUENCE_TARGET):	
	'''
		Fetches validation set and retuns as a numpy array
		args:

		returns: 
			features 	.:
			targets 	.: 
			lengths		.: 
			others		.: 
	'''		
	if not(ds_type in ['train', 'valid', 'test']):
		buff= 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
		raise ValueError(buff)
	else:
		if ds_type in ['train']:
			dataset_path= 	conf.DATASET_TRAIN_PATH 
			dataset_size=conf.DATASET_TRAIN_SIZE
		if ds_type in ['test']:
			dataset_path= 	conf.DATASET_TEST_PATH
			dataset_size=conf.DATASET_TEST_SIZE
		if ds_type in ['valid']:	
			dataset_path= 	conf.DATASET_VALID_PATH
			dataset_size=conf.DATASET_VALID_SIZE

	inputs, targets, lengths, others= tensor2numpy(
		dataset_path, 
		dataset_size,
		embeddings, 
		feat2size,
		input_sequence_features=input_features, 
		output_sequence_target=output_target, 
		msg='input_{:}: done converting {:} set to numpy'.format(ds_type, ds_type)
	)	

	return inputs, targets, lengths, others

def tensor2numpy(dataset_path, dataset_size, 
								embeddings, feat2size, input_sequence_features, 
								output_sequence_target,msg='tensor2numpy: done'):	
	'''
		Converts a .tfrecord into a numpy representation of a tensor
		args:

		returns: 
			features 	.:
			targets 	.: 
			lengths		.: 
			others		.: 
	'''	
	# other_features=['ARG_0', 'P', 'IDX', 'FUNC']
	# input_sequence_features=list(set(conf.SEQUENCE_FEATURES).intersection(set(filter_features)) - set(other_features)- set(['targets']))


	with tf.name_scope('pipeline'):
		X, T, L, D= input_fn(
			[dataset_path], 
			dataset_size, 
			1, 
			embeddings, 
			feat2size, 
			input_sequence_features=input_sequence_features,
			output_sequence_target=output_sequence_target
		)

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
				inputs, targets, times, descriptors=session.run([X, T, L, D])					

		except tf.errors.OutOfRangeError:
			print(msg)			

		finally:
			#When done, ask threads to stop
			coord.request_stop()			
			coord.join(threads)
	return inputs, targets, times, descriptors		


# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, batch_size,  num_epochs, 
						embeddings, feat2size, 
						input_sequence_features=conf.DEFAULT_INPUT_SEQUENCE_FEATURES, 
						output_sequence_target= conf.DEFAULT_OUTPUT_SEQUENCE_TARGET):
	'''
		Produces sequence_examples shuffling at every epoch while batching every batch_size
			number of examples
		
		args:
			filenames								.: list containing tfrecord file names
			batch_size 							.: integer containing batch size		
			num_epochs 							.: integer # of repetitions per example
			embeddings 							.: matrix [VOCAB_SZ, EMBEDDINGS_SZ] pre trained array
			feat2size   						.: int 
			input_sequence_features .: list containing the feature fields from .csv file

		returns:
			X_batch						.:
			T_batch 						.:
			L_batch						.: 
			D_batch				.: [M] features that serve as descriptors but are not used for training
					M= len(input_sequence_features)
	'''
	
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
	
	

	context_features, sequence_features= _read_and_decode(filename_queue)	

	X, T, L, D= _process(context_features, 
		sequence_features, 
		input_sequence_features, 
		output_sequence_target, 
		feat2size, 
		embeddings
	)	
	
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size

	# https://www.tensorflow.org/api_docs/python/tf/train/batch
	X_batch, T_batch, L_batch, D_batch =tf.train.batch(
		[X, T, L, D], 
		batch_size=batch_size, 
		capacity=capacity, 
		dynamic_pad=True
	)
	return X_batch, T_batch, L_batch, D_batch
		

def _read_and_decode(filename_queue):
	'''
		Decodes a serialized .tfrecords containing sequences
		args
			filename_queue.: list of strings containing file names which are added to queue

		returns
			context_features.: features that are held constant thru sequence ex: time, sequence id

			sequence_features.: features that are held variable thru sequence ex: word_idx

	'''
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# a serialized sequence example contains:
	# *context_features.: which are hold constant along the whole sequence
	#   	ex.: sequence_length
	# *sequence_features.: features that change over sequence 
	context_features, sequence_features= tf.parse_single_sequence_example(
		serialized_example,
		context_features=TF_CONTEXT_FEATURES,
		sequence_features=TF_SEQUENCE_FEATURES
	)

	return context_features, sequence_features


# conf.SEQUENCE_FEATURES=['IDX', 'P_S', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
# TARGET_FEATURE=['ARG_1']
def _process(context_features, sequence_features, 
							input_sequence_features, output_sequence_targets, 
							feat2size, embeddings):

	'''
		Maps context_features and sequence_features making embedding replacement as necessary

		args:
			context_features 				.: protobuffer containing features hold constant for example 
			sequence_features 			.: protobuffer containing features that change wrt time 
			input_sequence_features .: list of sequence features to be used as inputs
			embeddings        			.: matrix [EMBEDDING_SIZE, VOCAB_SIZE] containing pre computed word embeddings
			klass_size, 		  			.: int number of output classes
			

		returns:	
			X 									.: 
			T 						.: 
			L 								.: 
			D				.:

			2018-02-26: input_sequence_features introduced now client may select input fields
				(before it was hard coded and alll features from .csv where used)
	'''
	context_inputs=[]
	sequence_inputs=[]
	# those are not to be used as inputs but appear in the description of the data
	sequence_descriptors=[] 	
	sequence_target=[]

	# Fetch only context variable the length of the proposition
	L = context_features['L']

	sel = 	input_sequence_features +  [output_sequence_targets]
	#Read all inputs as tf.int64			
	#paginates over all available columnx	
	for key in conf.SEQUENCE_FEATURES:
		dense_tensor= tf.sparse_tensor_to_dense(sequence_features[key])		
		
		#Selects how to handle column from conf.META
		if key in sel: 
			if conf.META[key] in ['txt']: 
				dense_tensor1= tf.nn.embedding_lookup(embeddings, dense_tensor)
				
				
			elif conf.META[key] in ['hot']:
				dense_tensor1= tf.one_hot(
					dense_tensor, 
					feat2size[key],
					on_value=1,
					off_value=0,
					dtype=tf.int32
				)								
				if key in input_sequence_features:
					dense_tensor1= tf.cast(dense_tensor1, tf.float32)
				

			else: 
				if key in input_sequence_features:
					# Cast to tf.float32 in order to concatenate in a single array with embeddings					
					dense_tensor1=tf.expand_dims(tf.cast(dense_tensor,tf.float32), 2)
					
				else:
					dense_tensor1= dense_tensor
		else:
			#keep their numerical values 
			dense_tensor1= dense_tensor



		if key in input_sequence_features:
			sequence_inputs.append(dense_tensor1)
		elif key in [output_sequence_targets]:		
			T= tf.squeeze(dense_tensor1, 1, name='squeeze_T')
		else:
			sequence_descriptors.append(dense_tensor1)

	#UNCOMMENT
	X= tf.squeeze( tf.concat(sequence_inputs, 2),1, name='squeeze_X') 
	D= tf.concat(sequence_descriptors, 1)
	return X, T, L, D 

############################# tfrecords writer ############################# 
def tfrecords_builder(propbank_iter, dataset_type, lang='pt'):
	'''
		Iterates within propbank and saves records
	'''
	if not(dataset_type in ['train', 'valid', 'test']):
		buff= 'dataset_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(dataset_type)
		raise ValueError(buff)
	else:
		if dataset_type in ['train']:		
			total_propositions= conf.DATASET_TRAIN_SIZE 
		if dataset_type in ['valid']:		
			total_propositions= conf.DATASET_VALID_SIZE 
		if dataset_type in ['test']:
			total_propositions= conf.DATASET_TEST_SIZE  		

	tfrecords_path= conf.INPUT_DIR + 'db{:}_{:}_v2.tfrecords'.format(dataset_type,lang)	
	with open(tfrecords_path, 'w+') as f:
		writer= tf.python_io.TFRecordWriter(f.name)	
	
		l=1
		ex = None 
		prev_p = -1 
		helper_d= {} # that's a helper dict in order to abbreviate
		num_records=0
		num_propositions=0
		for index, d in propbank_iter:
			if d['P'] != prev_p:					
				if ex: 
					# compute the context feature 'L'
					ex.context.feature['L'].int64_list.value.append(l)			
					writer.write(ex.SerializeToString())        
				ex= tf.train.SequenceExample()
				l=1
				helper_d= {}
				num_propositions+=1
			else:
				l+=1

			for feat, value in d.items():				
				if not(feat in helper_d):
					helper_d[feat]=ex.feature_lists.feature_list[feat] 
				
				helper_d[feat].feature.add().int64_list.value.append(value)

			num_records+=1	
			prev_p=d['P']
			if num_propositions % 25: 
				msg= 'Processing {:}\trecords:{:5d}\tpropositions:{:5d}\tcomplete:{:0.2f}%\r'.format(
						dataset_type, num_records, num_propositions, 100*float(num_propositions)/total_propositions)        
				sys.stdout.write(msg)
				sys.stdout.flush() 	
			

		msg= 'Processing {:}\trecords:{:5d}\tpropositions:{:5d}\tcomplete:{:0.2f}%\n'.format(
					dataset_type, num_records, num_propositions, 100*float(num_propositions)/total_propositions)        
		sys.stdout.write(msg)
		sys.stdout.flush() 	
		

		# write the one last example		
		ex.context.feature['L'].int64_list.value.append(l)			
		writer.write(ex.SerializeToString())        			
		
	writer.close()
	print('Wrote to {:} found {:} propositions'.format(f.name, num_propositions))			

# if __name__== '__main__':
	# propbank= Propbank.recover('./datasets/binaries/db_pt_LEMMA_glove_s50.pickle')
	# UNCOMMENT this to save an updated version
	# propbank= Propbank()
	# propbank.define()
	# propbank.persist('')
	
	# for ds_type in ['train', 'test', 'valid']:
	# 	tfrecords_builder(propbank.iterator(ds_type), ds_type)
	
	# hotencode2sz= {feat: propbank.size(feat)
	# 		for feat, feat_type in conf.META.items() if feat_type == 'hot'}		
	

	# TEST			
	# test_sequence_features= ['ID', 'LEMMA']
	# X_valid, T_valid, L_valid, D_valid= tfrecords_extract(
	# 	'valid', 
	# 	propbank.embeddings, 
	# 	hotencode2sz, 
	# 	input_features=test_sequence_features,
	# 	output_target='T'
	# )
	
	# l1 = L_valid[0]
	# X1 = X_valid[0,:l1,:]
	# index = D_valid[0,:l1,0]
	# i0=0
	# import code; code.interact(local=dict(globals(), **locals()))			
	# # print('index:{:}'.format(index))
	# for j, feature in enumerate(test_sequence_features):				
	# 	for i, idx in  enumerate(index):
	# 		x1 = X1[i,:]
	# 		db_value= propbank.decode(propbank.data[feature][idx], feature)			
	# 		if conf.META[feature] in ['txt']:
	# 			# search for the index
	# 			min_dist=99999999
	# 			min_idx=-1				
	# 			for k in range(propbank.embeddings.shape[0]): 
	# 				check= propbank.embeddings[k]
	# 				norm= np.linalg.norm(check - x1[i0:i0+50])
	# 				if norm < min_dist:
	# 					min_dist=norm
	# 					min_idx= k
	# 			tf_value= propbank.idx2tok[min_idx]
	# 			print('min distance:', min_dist)
	# 		if conf.META[feature] in ['int']:
	# 			tf_value= int(x1[i0])

	# 		if conf.META[feature] in ['hot']:	
	# 			tf_value= np.argmax(x1[i0:i0+hotencode2sz[feature]])				

	# 		print('INDEX:{:}\tFEATURE:{:}\tDB:{:}\tTF{:}\t'.format(idx, feature, db_value, tf_value))				

	# 	if conf.META[feature] in ['txt']:
	# 		i0+= 50
	# 	else:
	# 		if conf.META[feature] in ['int']:
	# 			i0+=1
	# 		elif conf.META[feature] in ['hot']:
	# 				i0+= hotencode2sz[feature]			
	
	





