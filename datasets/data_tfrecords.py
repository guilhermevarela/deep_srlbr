'''
Created on Jan 25, 2018
	@author: Varela

	Generates and reads tfrecords
		* Generates train, valid, test datasets
		* Provides input_fn that acts as a feeder


	
	2018-02-21: added input_fn
	2018-02-26: added input_sequence_features to input_fn
'''
import pandas as pd 
import numpy as np 

#Uncomment if launched from /datasets
from data_propbankbr import  propbankbr_lazyload
from data_vocabularies import vocab_lazyload_with_embeddings, vocab_lazyload, vocab_preprocess

import tensorflow as tf 

EMBEDDING_PATH='datasets/embeddings/'
TARGET_PATH='datasets/inputs/01/'

#Must be both 1) inputs to the model 2) have a string representation
DEFAULT_SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC']
CTX_P_SEQUENCE_FEATURES=['CTX_P-3','CTX_P-2','CTX_P-1', 'CTX_P+1','CTX_P+2','CTX_P+3']
TARGET_FEATURE=['ARG_1']
EMBEDDABLE_FEATURES=['FORM','LEMMA', 'PRED']+ CTX_P_SEQUENCE_FEATURES


SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0'] + CTX_P_SEQUENCE_FEATURES + ['targets']

TF_SEQUENCE_FEATURES= {key:tf.VarLenFeature(tf.int64) 
	for key in SEQUENCE_FEATURES
}

TF_CONTEXT_FEATURES=	{
	'T': tf.FixedLenFeature([], tf.int64)			
}


############################# tfrecords reader ############################# 
# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, batch_size,  num_epochs, embeddings, klass_size, 
	input_sequence_features=DEFAULT_SEQUENCE_FEATURES):
	'''
		Produces sequence_examples shuffling at every epoch while batching every batch_size
			number of examples
		
		args:
			filenames								.: list containing tfrecord file names
			batch_size 							.: integer containing batch size		
			num_epochs 							.: integer # of repetitions per example
			embeddings 							.: matrix [VOCAB_SZ, EMBEDDINGS_SZ] pre trained array
			klass_size  						.: int 
			input_sequence_features .: list containing the feature fields from .csv file

		returns:
			input_batch 						.:
			target_batch 						.:
			length_batch						.: 
			descriptor_batch				.: [M] features that serve as descriptors but are not used for training
					M= len(input_sequence_features)
	'''
	
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	context_features, sequence_features= _read_and_decode(filename_queue)	

	inputs, targets, length, descriptors= _process(context_features, sequence_features, embeddings, klass_size, input_sequence_features=input_sequence_features)	
	
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size

	# https://www.tensorflow.org/api_docs/python/tf/train/batch
	input_batch, target_batch, length_batch, desc_batch =tf.train.batch(
		[inputs, targets, length, descriptors], 
		batch_size=batch_size, 

		capacity=capacity, 
		dynamic_pad=True
	)
	return input_batch, target_batch, length_batch, desc_batch

def input_sz(input_sequence_features, embedding_sz):	
	'''
		Returns input feature size 
		args:
			input_sequence_features	.: list of features being used as input

			embedding_sz .: int 	embedding layer dimensions
		returns:
			feature_sz  .: int feature sz
	'''
	feature_set=set(input_sequence_features)
	embeddable_set= set(EMBEDDABLE_FEATURES)
	feature_sz= len(feature_set-embeddable_set)
	feature_sz+=len(embeddable_set.intersection(feature_set))*embedding_sz
	return feature_sz
		

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


# SEQUENCE_FEATURES=['IDX', 'P_S', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
# TARGET_FEATURE=['ARG_1']
def _process(context_features, sequence_features, embeddings, 
		klass_size, input_sequence_features):

	'''
		Maps context_features and sequence_features making embedding replacement as necessary

		args:
			context_features 				.: protobuffer containing features hold constant for example 
			sequence_features 			.: protobuffer containing features that change wrt time 
			embeddings        			.: matrix [EMBEDDING_SIZE, VOCAB_SIZE] containing pre computed word embeddings
			klass_size, 		  			.: int number of output classes
			input_sequence_features .: list of sequence features to be used as inputs

		returns:	
			inputs 									.: 
			descriptors 						.: 
			targets 								.: 
			sequence_length 				.:

			2018-02-26: input_sequence_features introduced now client may select input fields
				(before it was hard coded and alll features from .csv where used)
	'''
	context_inputs=[]
	sequence_inputs=[]
	# those are not to be used as inputs but appear in the description of the data
	sequence_descriptors=[] 	
	sequence_target=[]

	context_keys=['T']
	for key in context_keys:
		val32= tf.cast(context_features[key], tf.int32)
		context_inputs.append( val32	 )

	#Read all inputs as tf.int64		
	for key in SEQUENCE_FEATURES:
		dense_tensor= tf.sparse_tensor_to_dense(sequence_features[key])		
		# import code; code.interact(local=dict(globals(), **locals()))		
		if key in input_sequence_features:
			if key in EMBEDDABLE_FEATURES: # ['PRED', 'LEMMA', 'CTX_P-3','CTX_P-2','CTX_P-1', 'CTX_P+1','CTX_P+2','CTX_P+3']
				dense_tensor1= tf.nn.embedding_lookup(embeddings, dense_tensor)
				sequence_inputs.append(dense_tensor1)
			else: #key in ['ID', 'M_R']: # numeric inputs
				# Cast to tf.float32 in order to concatenate in a single array with embeddings
				dense_tensor1=tf.expand_dims(tf.cast(dense_tensor,tf.float32), 2)
				sequence_inputs.append(dense_tensor1)
		elif key in ['targets']:			
			dense_tensor1= tf.one_hot(
				dense_tensor, 
				klass_size,
				on_value=1,
				off_value=0,
				dtype=tf.int32
			)
			Y= tf.squeeze(dense_tensor1,1, name='squeeze_Y')
		else: # descriptors
			sequence_descriptors.append(dense_tensor)

			
	X= tf.squeeze( tf.concat(sequence_inputs, 2),1, name='squeeze_X') 
	D= tf.concat(sequence_descriptors, 1)
	return X, Y, context_inputs[0], D 

############################# tfrecords writer ############################# 

def proposition2sequence_example(
	dict_propositions, dict_vocabs, sequence_features=SEQUENCE_FEATURES, target_feature=TARGET_FEATURE):
	'''
		Maps a propbank proposition into a sequence example - producing a minibatch
	'''
	ex= tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length=len(dict_propositions[target_feature[0]])
	ex.context.feature['T'].int64_list.value.append(sequence_length)

	
	#Make a dictionary of feature_lists
	sequence_dict={}
	for key in sequence_features:
		sequence_dict[key]= ex.feature_lists.feature_list[key]
		if key in dict_propositions:
			for token in dict_propositions[key]:					
				# if token is a string lookup in a dict
				if isinstance(token, str):				
					idx= get_idx(token, key, dict_vocabs)
					sequence_dict[key].feature.add().int64_list.value.append(idx)
				else:								
					sequence_dict[key].feature.add().int64_list.value.append(token)

	f1_targets= ex.feature_lists.feature_list['targets']
	for key in target_feature:		
		for token in dict_propositions[key]:		
			idx= get_idx(token, key, dict_vocabs)						
			f1_targets.feature.add().int64_list.value.append(idx)
	
	return ex	
	
def df2data_dict(df):	
	return df.to_dict(orient='list')

def make_dict_vocabs(df):
	'''
		Looks up dtypes from df which are strings, lazyloads then
		args:
			df .: pandas dataframe containing SEQUENCE_FEATURES and TARGET_FEATURE

		returns: 
			dict_vocabs .: a dictionary of vocabularies the 
				keys in SEQUENCE_FEATURES or TARGET_FEATURE

	'''
	dict_vocabs ={} 
	features= set(SEQUENCE_FEATURES + TARGET_FEATURE)

	df_features= df.dtypes.index.tolist()
	dtypes_features=df.dtypes.values.tolist()


	selection_features= [df_features[i]
		for i, val in enumerate(dtypes_features) if str(val) =='object']

	features = features.intersection(set(selection_features))			
	for feat in features:
		if isembeddable(feat): # Embeddings features
			if not('word2idx' in dict_vocabs):
				dict_vocabs['word2idx'], _ =vocab_lazyload_with_embeddings('LEMMA', input_dir=TARGET_PATH) 
		else:
			if not(feat in dict_vocabs):
				dict_vocabs[feat] =vocab_lazyload(feat)

	return dict_vocabs			

def isembeddable(key):
	'''
		True if key has a string representation and belongs to the input
		args:
			key .:	string representing a valid field		
		
		returns:
			bool
	'''
	return key in EMBEDDABLE_FEATURES

def get_idx(token, key, dict_vocabs):
	'''
		Looks up dict_vocabs for token
		args:
			token .: string representing a stringyfied token
							ex .: (A0*, A0, 'estar'
			key 	.: fstring representing eature in the dataset belonging to 
				inputs, targers, descriptors, mini_batches

			dict_vocabs .: dictionary of dicionaries 
				outer most .: features
				inner most .: value within the feature 	

		returns:
			idx .: int indexed token
	'''
	if isembeddable(key):
		this_vocab= dict_vocabs['word2idx']
		try:
			idx = this_vocab[vocab_preprocess(token)]
		except KeyError:
			idx= 0 
	else:
		idx = dict_vocabs[key][token]
	return idx

if __name__== '__main__':
	df=propbankbr_lazyload('zhou')	
	
	dict_vocabs= make_dict_vocabs(df) # makes dictionary using tokens from whole dataset
	for dstype in ['train', 'valid', 'test']:
		tfrecords_path= '{}{}.tfrecords'.format(TARGET_PATH, dstype)

		
		df=propbankbr_lazyload('zhou_{}'.format(dstype))	
		p0 = min(df['P'])
		pn = max(df['P'])	# number of propositions		

		with open(tfrecords_path, 'w+') as f:
			writer= tf.python_io.TFRecordWriter(f.name)

			for p in range(p0, pn+1):
				df_prop= df[df['P']==p]			
				ex= proposition2sequence_example(
					df2data_dict( df_prop ), 
					dict_vocabs
				)	
				writer.write(ex.SerializeToString())        
	    
			writer.close()
			print('Wrote to {} found {} propositions'.format(f.name, pn-p0+1))		





