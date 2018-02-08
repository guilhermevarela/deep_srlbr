'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Provides file input and output utility functions
		* Feed a input pipeline 
		* Save and restore model
		* Download embeddings
'''

import tensorflow as tf 

import re 
import glob
import os.path


from datasets.data_vocabularies import vocab_lazyload_with_embeddings, vocab_lazyload  

DEFAULT_KLASS_SIZE=22

# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, batch_size,  num_epochs, embeddings, klass_size=DEFAULT_KLASS_SIZE):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	context_features, sequence_features= _read_and_decode(filename_queue)	

	inputs, targets, length= _process(context_features, sequence_features, embeddings, klass_size)	
	
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size

	# https://www.tensorflow.org/api_docs/python/tf/train/batch
	input_batch, target_batch, length_batch=tf.train.batch(
		[inputs, targets, length], 
		batch_size=batch_size, 
		capacity=capacity, 
		dynamic_pad=True
	)
	return input_batch, target_batch, length_batch

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
		context_features={
			'T': tf.FixedLenFeature([], tf.int64)			
		},
		sequence_features={
			'ID':tf.VarLenFeature(tf.int64),			
			'PRED':tf.VarLenFeature(tf.int64),			
			'LEMMA': tf.VarLenFeature(tf.int64),
			'M_R':tf.VarLenFeature(tf.int64),
			'targets':tf.VarLenFeature(tf.int64)		
		}
	)

	return context_features, sequence_features



def _process(context_features, sequence_features, embeddings, klass_size=DEFAULT_KLASS_SIZE):
	context_inputs=[]
	sequence_inputs=[]
	sequence_target=[]

	context_keys=['T']
	for key in context_keys:
		val32= tf.cast(context_features[key], tf.int32)
		context_inputs.append( val32	 )

	#Read all inputs as tf.int64	
	sequence_keys=['ID', 'M_R', 'PRED', 'LEMMA', 'targets']
	for key in sequence_keys:
		dense_tensor= tf.sparse_tensor_to_dense(sequence_features[key])
		if key in ['PRED', 'LEMMA']:
			dense_tensor1= tf.nn.embedding_lookup(embeddings, dense_tensor)
			sequence_inputs.append(dense_tensor1)

		# Cast to tf.float32 in order to concatenate in a single array with embeddings
		if key in ['ID', 'M_R']:
			dense_tensor1=tf.expand_dims(tf.cast(dense_tensor,tf.float32), 2)
			sequence_inputs.append(dense_tensor1)

		if key in ['targets']:			
			dense_tensor1= tf.one_hot(
				dense_tensor, 
				klass_size,
				on_value=1,
				off_value=0,
				dtype=tf.int32
			)

			Y= tf.squeeze(dense_tensor1,1, name='squeeze_Y' )
			

	
	X= tf.squeeze( tf.concat(sequence_inputs, 2),1, name='squeeze_X') 
	return X, Y, context_inputs[0]

def mapper_get(column_in, column_out, input_dir):
	'''
		Returns idx2values mappings (dicts) and embeddings (np.ndarray)

		args:
			column_out.:string	representing the column with outputs
				valid arguments are ARG, ARG_Y

			column_in .: string representing the column with embeddings 
				valid arguments are LEMMA, FORM, PRED

			input_dir  .: string containing inputs to be read 
				ex: datasets/inputs/00 

		returns:
			klass2idx .: dict mapping classes (str) to idx 

			word2idx .:  dict mapping words (str) to idx 

			embeddings .: embeddings
	'''	
	word2idx,  embeddings= vocab_lazyload_with_embeddings(column_in, input_dir=input_dir)		
	
	klass2idx = vocab_lazyload(column_out, input_dir=input_dir)		

	return klass2idx, word2idx, embeddings 

def dir_getmodels(lr, hidden_sizes, model_name='multi_bibasic_lstm'):	
	'''
		Makes a directory name for models from hyperparams

		args:
			lr  .: float learning rate
			
			hidden_sizes .:  list of ints

			model_name .:  string represeting the model
		
		returns:
			experiment_dir .:  string representing a valid relative path
					format 'logs/model_name/hparams/dd'

	'''
	prefix= 'models/' + model_name
	hparam_string= _make_hparam_string(lr, hidden_sizes)
	return _make_dir(prefix, hparam_string)

def dir_getlogs(lr, hidden_sizes ,model_name='multi_bibasic_lstm'):
	'''
		Makes a directory name for logs from hyperparams

		args:
			lr  .: float learning rate
			
			hidden_sizes .:  list of ints

			model_name .:  string represeting the model
		
		returns:
			experiment_dir .:  string representing a valid relative path
					format 'logs/model_name/hparams/dd'

	'''
	prefix= 'logs/' + model_name
	hparam_string= _make_hparam_string(lr, hidden_sizes)
	return _make_dir(prefix, hparam_string)

def _make_hparam_string(lr, hidden_sizes):
	'''
		Makes a directory name from hyper params

		args:
			lr  .: float learning rate
			
			hidden_sizes .:  list of ints
		
		returns:
			experiment_dir .:  string representing a valid relative path

	'''
	hparam_string= 'lr{:.2e}'.format(float(lr))

	hparam_string+= ',hs%s' % re.sub(r', ','x', re.sub(r'\[|\]','',str(hidden_sizes)))
	return hparam_string

def _make_dir(prefix, hparam_string):
	'''
		Creates a path by incrementing the number of experiments under prefix/hparam_string
		args:
			prefix string .:

		returns:
			experiment_dir string .:
	'''
	experiment_dir= prefix + '/' + hparam_string + '/'
	experiment_ids= sorted(glob.glob(experiment_dir + '[0-9]*'))	
	if len(experiment_ids)==0:
		experiment_dir+= '00/'
	else:
		experiment_dir+= '{:02d}/'.format(int(re.sub(experiment_dir,'',experiment_ids[-1]))+1)

	if not(os.path.exists(experiment_dir)):		
		os.makedirs(experiment_dir)	

	return experiment_dir





	