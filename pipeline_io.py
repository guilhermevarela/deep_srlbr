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





	