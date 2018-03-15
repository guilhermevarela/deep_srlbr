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


EMBEDDING_PATH='datasets/embeddings/'
# TARGET_PATH='datasets/inputs/01/'
# TARGET_PATH='datasets/inputs/02/'
TARGET_PATH='datasets/inputs/03/'
dataset_train= TARGET_PATH + 'dbtrain_pt.tfrecords'
dataset_valid= TARGET_PATH + 'dbvalid_pt.tfrecords'
dataset_tesgt= TARGET_PATH + 'dbtest_pt.tfrecords'

#deprecate this shit
#Must be both 1) inputs to the model 2) have a string representation
# DEFAULT_SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'GPOS', 'M_R', 'PRED', 'FUNC']
# CTX_P_SEQUENCE_FEATURES=['CTX_P-3','CTX_P-2','CTX_P-1', 'CTX_P+1','CTX_P+2','CTX_P+3']
# TARGET_FEATURE=['ARG_1']
# EMBEDDABLE_FEATURES=['FORM','LEMMA', 'PRED']+ CTX_P_SEQUENCE_FEATURES


# conf.SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0'] + CTX_P_SEQUENCE_FEATURES + ['targets']
# Language model
# conf.SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'GPOS', 'M_R', 'PRED', 'FUNC', 'ARG_0'] + CTX_P_SEQUENCE_FEATURES + ['targets']

# conf.TF_SEQUENCE_FEATURES= {key:tf.VarLenFeature(tf.int64) 
# 	for key in conf.SEQUENCE_FEATURES
# }

# conf.TF_CONTEXT_FEATURES=	{
# 	'T': tf.FixedLenFeature([], tf.int64)			
# }

# conf.DATASET_VALID_SIZE= 569
# conf.DATASET_TEST_SIZE= 5099
# DATASET_TEST_SIZE=  263



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
			dataset_path= 	dataset_train
			dataset_size=conf.DATASET_TRAIN_SIZE
		if ds_type in ['test']:
			dataset_path= 	dataset_test
			dataset_size=conf.DATASET_TEST_SIZE
		if ds_type in ['valid']:	
			dataset_path= 	dataset_valid
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

# def input_train(embeddings, feat2size, filter_features=input_sequence_features):	
	
# 	inputs, targets, lengths, others= tensor2numpy(
# 		dataset_train, 
# 		conf.DATASET_TEST_SIZE,
# 		embeddings, 
# 		feat2size,		
# 		filter_features=filter_features, 
# 		msg='input_train: done converting train set to numpy'
# 	)	
	
# 	return inputs, targets, lengths, others

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
				import code; code.interact(local=dict(globals(), **locals()))			
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

# def input_sz(input_sequence_features, embedding_sz):	
# 	'''
# 		Returns input feature size 
# 		args:
# 			input_sequence_features	.: list of features being used as input

# 			embedding_sz .: int 	embedding layer dimensions
# 		returns:
# 			feature_sz  .: int feature sz
# 	'''
# 	feature_set=set(input_sequence_features)
# 	embeddable_set= set(EMBEDDABLE_FEATURES)
# 	feature_sz= len(feature_set-embeddable_set)
# 	# feature_sz+=len(embeddable_set.intersection(feature_set))*embedding_sz+25
# 	# GPOS
# 	feature_sz+=len(embeddable_set.intersection(feature_set))*embedding_sz+25-1
# 	return feature_sz
		

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
					print('integer inputs:', key)
				else:
					dense_tensor1= dense_tensor
					print('integer outputs:', key)
		else:
			#keep their numerical values 
			dense_tensor1= dense_tensor
			print('integer descriptors:', key)



		if key in input_sequence_features:
			sequence_inputs.append(dense_tensor1)
		elif key in [output_sequence_targets]:		
			T= tf.squeeze(dense_tensor1, 1, name='squeeze_T')
		else:
			sequence_descriptors.append(dense_tensor1)
		
			

	# import code; code.interact(local=dict(globals(), **locals()))			
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

	tfrecords_path= TARGET_PATH + 'db{:}_{:}.tfrecords'.format(dataset_type,lang)	
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

# def proposition2sequence_example(
# 	dict_propositions, dict_vocabs, sequence_features=conf.SEQUENCE_FEATURES, target_feature=TARGET_FEATURE):
# 	'''
# 		Maps a propbank proposition into a sequence example - producing a minibatch
# 	'''
# 	ex= tf.train.SequenceExample()
# 	# A non-sequential feature of our example
# 	sequence_length=len(dict_propositions[target_feature[0]])
# 	ex.context.feature['T'].int64_list.value.append(sequence_length)

	
# 	#Make a dictionary of feature_lists
# 	sequence_dict={}
# 	for key in sequence_features:
# 		sequence_dict[key]= ex.feature_lists.feature_list[key]
# 		if key in dict_propositions:
# 			for token in dict_propositions[key]:					
# 				# if token is a string lookup in a dict

# 				if isinstance(token, str):				
# 					idx= get_idx(token, key, dict_vocabs)
# 					sequence_dict[key].feature.add().int64_list.value.append(idx)
# 				else:								
# 					sequence_dict[key].feature.add().int64_list.value.append(token)

# 	f1_targets= ex.feature_lists.feature_list['targets']
# 	for key in target_feature:		
# 		for token in dict_propositions[key]:		
# 			idx= get_idx(token, key, dict_vocabs)						
# 			f1_targets.feature.add().int64_list.value.append(idx)
	
# 	return ex	
	
# def df2data_dict(df):	
# 	return df.to_dict(orient='list')

# def make_dict_vocabs(df):
# 	'''
# 		Looks up dtypes from df which are strings, lazyloads then
# 		args:
# 			df .: pandas dataframe containing conf.SEQUENCE_FEATURES and TARGET_FEATURE

# 		returns: 
# 			dict_vocabs .: a dictionary of vocabularies the 
# 				keys in conf.SEQUENCE_FEATURES or TARGET_FEATURE

# 	'''
# 	dict_vocabs ={} 
# 	features= set(conf.SEQUENCE_FEATURES + TARGET_FEATURE)

# 	df_features= df.dtypes.index.tolist()
# 	dtypes_features=df.dtypes.values.tolist()


# 	selection_features= [df_features[i]
# 		for i, val in enumerate(dtypes_features) if str(val) =='object']

# 	features = features.intersection(set(selection_features))			
# 	for feat in features:
# 		if isembeddable(feat): # Embeddings features
# 			if not('word2idx' in dict_vocabs):
# 				dict_vocabs['word2idx'], _ =vocab_lazyload_with_embeddings('LEMMA', input_dir=TARGET_PATH) 
# 		else:
# 			if not(feat in dict_vocabs):
# 				dict_vocabs[feat] =vocab_lazyload(feat)

# 	return dict_vocabs			

# def isembeddable(key):
# 	'''
# 		True if key has a string representation and belongs to the input
# 		args:
# 			key .:	string representing a valid field		
		
# 		returns:
# 			bool
# 	'''
# 	return key in EMBEDDABLE_FEATURES

# def get_idx(token, key, dict_vocabs):
# 	'''
# 		Looks up dict_vocabs for token
# 		args:
# 			token .: string representing a stringyfied token
# 							ex .: (A0*, A0, 'estar'
# 			key 	.: fstring representing eature in the dataset belonging to 
# 				inputs, targers, descriptors, mini_batches

# 			dict_vocabs .: dictionary of dicionaries 
# 				outer most .: features
# 				inner most .: value within the feature 	

# 		returns:
# 			idx .: int indexed token
# 	'''
# 	if isembeddable(key):
# 		this_vocab= dict_vocabs['word2idx']
# 		try:
# 			idx= this_vocab[vocab_preprocess(token)]
# 		except KeyError:
# 			idx=  this_vocab[token.lower()] 
# 	else:
# 		idx = dict_vocabs[key][token]
# 	return idx

if __name__== '__main__':
	# df=propbankbr_lazyload('zhou_1')	
	
	# dict_vocabs= make_dict_vocabs(df) # makes dictionary using tokens from whole dataset
	#test tfrecords_builder
	propbank= Propbank.recover('db_pt_LEMMA_glove_s50.pickle')
	# propbank= Propbank()
	# propbank.define()
	# propbank.persist('')
	
	
	# for dstype in ['train', 'valid', 'test']:
		# tfrecords_builder(propbank.iterator(dstype), dstype)
	# tfrecords_builder(propbank.iterator('test'), 'test')	

	# tfrecords_path= '{}{}.tfrecords'.format(TARGET_PATH, dstype)

	
	# df=propbankbr_lazyload('zhou_1_{}'.format(dstype))	
	# p0 = min(df['P'])
	# pn = max(df['P'])	# number of propositions		

	# with open(tfrecords_path, 'w+') as f:
	# 	writer= tf.python_io.TFRecordWriter(f.name)

	# 	for p in range(p0, pn+1):
	# 		df_prop= df[df['P']==p]			
	# 		ex= proposition2sequence_example(
	# 			df2data_dict( df_prop ), 
	# 			dict_vocabs
	# 		)	
	# 		writer.write(ex.SerializeToString())        
    
	# 	writer.close()
		# print('Wrote to {} found {} propositions'.format(f.name, pn-p0+1))		
	# import code; code.interact(local=dict(globals(), **locals()))					
	hotencode2sz= {feat: propbank.size(feat)
			for feat, feat_type in conf.META.items() if feat_type == 'hot'}		
	


	X_train, T_train, L_train, D_train= tfrecords_extract('train', propbank.embeddings, hotencode2sz)
	# print(X_train.shape)
	# print(T_train.shape)
	# print(L_train.shape)
	# print(D_train.shape)
	import code; code.interact(local=dict(globals(), **locals()))			
	





