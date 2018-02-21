'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Provides file input and output utility functions
		* Feed a input pipeline 
		* Save and restore model
		* Download embeddings
'''

import tensorflow as tf 

import pandas as pd 
import re 
import glob
import os.path


from datasets.data_vocabularies import vocab_lazyload_with_embeddings, vocab_lazyload  

DEFAULT_KLASS_SIZE=22

SETTINGS=[
	'INPUT_PATH',
	'MODEL_NAME',
	'DATASET_TRAIN_SIZE',
	'DATASET_VALID_SIZE',
	'DATASET_TEST_SIZE',
	'lr',
	'reg',
	'HIDDEN_SIZE',
	'EMBEDDING_SIZE',
	'FEATURE_SIZE',
	'KLASS_SIZE',
	'BATCH_SIZE',
	'N_EPOCHS',
	'DISPLAY_STEP',
]
# EMBEDDABLE_FEATURES=['FORM','LEMMA', 'PRED']
# SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
# TARGET_FEATURE=['ARG_1']
# output_persist_Yhat(outputs_dir, descriptors_valid, Yhat_valid, mb_valid, klass2idx, 'Yhat_valid')
def output_persist_Yhat(output_dir, descriptors, Yhat, mb_sizes, idx2vocab, filename):
	'''
		Decodes predictions Yhat using idx2vocab and writes as a pandas DataFrame
		args
			output_dir .: string containing a valid dir to export tha settings
			
			idx .: list of ints holding the original indexes

			Yhat .: np.ndarray of ints 

			mb_sizes .: list with mini batch sizes

			idx2vocab:

			filename .: string representing the filename to save			
	'''
	if not(isinstance(Yhat,list)):
		l=Yhat.tolist()

	vocab2idx= {value:key for key, value in idx2vocab.items()}		
	#restore only the minibatch sizes and decode it
	tag_decoded =[vocab2idx[item] for i, sublist in enumerate(l) 
		for j, item in enumerate(sublist) if j < mb_sizes[i]  ]

	idx_decoded =[subsublist[0] for i, sublist in enumerate(idx.tolist()) 
		for j, subsublist in enumerate(sublist) if j < mb_sizes[i]]

	pred_decoded =[subsublist[0] for i, sublist in enumerate(predicate_idx.tolist()) 
		for j, subsublist in enumerate(sublist) if j < mb_sizes[i]]		
	
	prev_tag=''	
	
	this_tag=''
	new_tags=[]

	l= len(idx_decoded)	
	i=0
	for idx, pred, tag in zip(idx_decoded,pred_decoded,tag_decoded):
		#define left 
		if ((tag != prev_tag) and (tag != '*')): 
			this_tag= '(' + tag + '*'
		else:
			this_tag+= '*'

		#define right
		if (i<l-1): 
			if ((tag != tag_decoded[i+1]) and (tag != '*')):
				this_tag+= ')'


			if (pred == pred_decoded[i+1]):
				prev_tag= tag 
			else:
				prev_tag= ''	
		
		new_tags.append(this_tag)					
		this_tag= ''
		i+=1
	

	file_path= output_dir +  filename + '.csv'
		
	df_1= pd.DataFrame(data=tag_decoded, columns=['Y_0'], index=idx_decoded)
	df_2= pd.DataFrame(data=new_tags, columns=['Y_1'], index=idx_decoded)

	df= pd.concat((df_1,df_2), axis=1)
	df.index.column='IDX'	
	df.to_csv(file_path)

	

def output_persist_settings(output_dir, vars_dict, to_persist=SETTINGS):
	'''
		Writes on output_dir a settings.txt file with the settings
		args
			output_dir .: string containing a valid dir to export tha settings
			
			vars_dict .: dict with local / global variables from main scope

			to_persist .: list (optional) with the names of the variables to be persisted
		
		returns
			persisted_dict	.: dict with the subset of variables that were persisted
	'''
	#captures only the relevant variables
	settings_dict= {var_name: var_value
		for var_name, var_value in vars_dict.items() if var_name in to_persist}

	with open(output_dir + 'settings.txt','w+') as f:
		for var_name, var_value in settings_dict.items():		
			f.write('{}: {}\n'.format(var_name, var_value))
		f.close()
	return settings_dict


# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, batch_size,  num_epochs, embeddings, klass_size=DEFAULT_KLASS_SIZE):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	context_features, sequence_features= _read_and_decode(filename_queue)	

	inputs, targets, length, descriptors= _process(context_features, sequence_features, embeddings, klass_size)	
	
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
			'IDX':tf.VarLenFeature(tf.int64),			
			'P':tf.VarLenFeature(tf.int64),			
			'ID':tf.VarLenFeature(tf.int64),			
			'PRED':tf.VarLenFeature(tf.int64),			
			'LEMMA': tf.VarLenFeature(tf.int64),
			'M_R':tf.VarLenFeature(tf.int64),			
			'FUNC':tf.VarLenFeature(tf.int64),			
			'ARG_0':tf.VarLenFeature(tf.int64),			
			'targets':tf.VarLenFeature(tf.int64)		
		}
	)

	return context_features, sequence_features


# SEQUENCE_FEATURES=['IDX', 'P_S', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
# TARGET_FEATURE=['ARG_1']
def _process(context_features, sequence_features, embeddings, klass_size=DEFAULT_KLASS_SIZE):
	context_inputs=[]
	sequence_inputs=[]
	# those are not to be used as inputs but appear in the description of the data
	sequence_descriptors=[] 	
	sequence_target=[]

	context_keys=['T']
	for key in context_keys:
		val32= tf.cast(context_features[key], tf.int32)
		context_inputs.append( val32	 )

	# EMBEDDABLE_FEATURES=['FORM','LEMMA', 'PRED']
	# SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
	# TARGET_FEATURE=['ARG_1']
	#Read all inputs as tf.int64	
	sequence_keys=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0', 'targets']
	for key in sequence_keys:
		dense_tensor= tf.sparse_tensor_to_dense(sequence_features[key])		

		if key in ['IDX', 'P', 'FUNC','ARG_0']: # descriptors
			sequence_descriptors.append(dense_tensor)

		if key in ['PRED', 'LEMMA']: # embedded inputs
			dense_tensor1= tf.nn.embedding_lookup(embeddings, dense_tensor)
			sequence_inputs.append(dense_tensor1)

		# Cast to tf.float32 in order to concatenate in a single array with embeddings
		if key in ['ID', 'M_R']: # numeric inputs
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
			Y= tf.squeeze(dense_tensor1,1, name='squeeze_Y')
			

	
	X= tf.squeeze( tf.concat(sequence_inputs, 2),1, name='squeeze_X') 
	D= tf.concat(sequence_descriptors, 1)
	return X, Y, context_inputs[0], D 

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

def dir_getoutputs(lr, hidden_sizes, model_name='multi_bibasic_lstm'):	
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
	prefix= 'outputs/' + model_name
	hparam_string= _make_hparam_string(lr, hidden_sizes)
	return _make_dir(prefix, hparam_string)

# def dir_getmodels(lr, hidden_sizes, model_name='multi_bibasic_lstm'):	
# 	'''
# 		Makes a directory name for models from hyperparams

# 		args:
# 			lr  .: float learning rate
			
# 			hidden_sizes .:  list of ints

# 			model_name .:  string represeting the model
		
# 		returns:
# 			experiment_dir .:  string representing a valid relative path
# 					format 'logs/model_name/hparams/dd'

# 	'''
# 	prefix= 'models/' + model_name
# 	hparam_string= _make_hparam_string(lr, hidden_sizes)
# 	return _make_dir(prefix, hparam_string)

# def dir_getlogs(lr, hidden_sizes ,model_name='multi_bibasic_lstm'):
# 	'''
# 		Makes a directory name for logs from hyperparams

# 		args:
# 			lr  .: float learning rate
			
# 			hidden_sizes .:  list of ints

# 			model_name .:  string represeting the model
		
# 		returns:
# 			experiment_dir .:  string representing a valid relative path
# 					format 'logs/model_name/hparams/dd'

# 	'''
# 	prefix= 'logs/' + model_name
# 	hparam_string= _make_hparam_string(lr, hidden_sizes)
# 	return _make_dir(prefix, hparam_string)

def _make_hparam_string(lr, hidden_sizes):
	'''
		Makes a directory name from hyper params

		args:
			lr  .: float learning rate
			
			hidden_sizes .:  list of ints
		
		returns:
			experiment_dir .:  string representing a valid relative path

	'''
	
	hs=re.sub(r', ','x', re.sub(r'\[|\]','',str(hidden_sizes)))
	hparam_string= 'lr{:.2e}_hs{:}'.format(float(lr),hs)	
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





	