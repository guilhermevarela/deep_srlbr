'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Provides file input and output utility functions
		* Feed a input pipeline 
		* Save and restore model
		* Download embeddings
'''
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
def outputs_predictions_persist(
	output_dir, indexes, predicates, predictions, batch_sizes, vocab2idx, filename):
	'''
		Decodes predictions using vocab2idx and writes as a pandas DataFrame

		args
			output_dir 	.: string containing a valid dir to export tha settings
			indexes 		.: int matrix   [BATCH_SIZE, MAX_TIME] holding original indexes
			predicates 	.: int matrix   [BATCH_SIZE, MAX_TIME] holding original predicate index
			predictions .: int matrix   [BATCH_SIZE, MAX_TIME] 
			batch_sizes .: list with mini batch sizes [BATCH_SIZE] 
			vocab2idx		.:

			filename .: string representing the filename to save			
	'''

	if not(isinstance(predictions,list)):
		l=predictions.tolist()

	
	idx2vocab= {value:key for key, value in vocab2idx.items()}		
	#restore only the minibatch sizes and decode it
	tag_decoded =[idx2vocab[item] for i, sublist in enumerate(l) 
		for j, item in enumerate(sublist) if j < batch_sizes[i]  ]


	idx_decoded =[idx for i, sublist in enumerate(indexes.tolist())
		for idx in sublist[:batch_sizes[i]]] 
		

	pred_decoded =[pred for i, sublist in enumerate(predicates.tolist())
		for pred in sublist[:batch_sizes[i]]] 
	
	prev_tag=''	
	
	this_tag=''
	new_tags=[]

	l= len(idx_decoded)	
	i=0
	for idx, pred, tag in zip(idx, pred_decoded,tag_decoded):
		#define left 
		if ((tag != prev_tag) and (tag != '*')): 
			this_tag= '(' + tag + '*'
		else:
			this_tag+= '*'

		#define right
		import code; code.interact(local=dict(globals(), **locals()))		
		if (i<l-1): 
			if (pred == pred_decoded[i+1]):			
				if ((tag != tag_decoded[i+1]) and (tag != '*')):
					this_tag+= ')'	
					prev_tag= tag 
			else:
				if (new_tags[-1] != '*)' and new_tags[-1] != '*'): # FIX CLOSING TAGS
					new_tags[-1]= '*)'	
				this_tag= '*'
				prev_tag= tag 

			# if ((tag != tag_decoded[i+1]) and (tag != '*')):
			# 	this_tag+= ')'

			# if (pred == pred_decoded[i+1]):
			# 	prev_tag= tag 
			# else:
			# 	prev_tag= ''	
		
		new_tags.append(this_tag)					
		this_tag= ''
		i+=1
	

	file_path= output_dir +  filename + '.csv'		
	columns=['IDX','Y_0', 'Y_1']
	data=[idx_decoded, new_tags, tag_decoded]
	df=pd.DataFrame.from_dict(dict(zip(columns,data))).set_index('IDX', inplace=False)	
	df.to_csv(file_path)

def outputs_settings_persist(output_dir, vars_dict, to_persist=SETTINGS):
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

def outputs_conll(df, target_column='Y_0'):
	'''
		Converts a dataset to conll format 
	'''
	df= df[['S', 'P', 'FUNC', target_column]]

	#Replace '-' making it easier to concatenate
	sub_fn= lambda x: re.sub('-', '', x)
	df['FUNC'] = df['FUNC'].apply(sub_fn)


	s0= min(df['S'])
	sn= max(df['S'])
	for i,si in enumerate(range(s0,sn+1)):
	    dfsi=df[ df['S'] == si ]
	    p0= min(dfsi['P'])
	    pn= max(dfsi['P'])    
	    for j,p in enumerate(range(p0,pn+1)): # concatenate by argument columns
	        dfpj=dfsi[dfsi['P']==p].reset_index(drop=True)
	        if j==0:
	            dfp=dfpj[['FUNC','Y_0']]
	            dfp=dfp.rename(columns={'Y_0': 'ARG0'})
	        else:
	            dfp['FUNC']=dfp['FUNC'].map(str).values + dfpj['FUNC'].map(str).values
	            dfp= pd.concat((dfp, dfpj['Y_0']), axis=1)
	            dfp=dfp.rename(columns={'Y_0': 'ARG{:d}'.format(p-p0)})            
	    if i==0:        
	        dfconll= dfp
	    else:
	        dfconll= dfconll.append(dfp).fillna('')
	        
	#Replace '-' making it easier to concatenate
	sub_fn= lambda x: '-' if len(x) == 0 else x
	dfconll['FUNC'] = dfconll['FUNC'].apply(sub_fn)

	#Order columns to fit conll standard
	num_columns=dfconll.columns
	usecolumns=['FUNC'] + ['ARG{:d}'.format(i) for i in range(num_columns-1)]
	return dfconll[usecolumns]
	

def mapper_get(column_in, column_out, input_dir):
	'''
		Returns idx2values mappings (dicts) and embeddings (np.ndarray)

		args:
			column_in .: string representing the column with embeddings 
				valid arguments are LEMMA, FORM, PRED

			column_out.:string	representing the column with outputs
				valid arguments are ARG, ARG_Y

			input_dir  .: string containing inputs to be read 
				ex: datasets/inputs/00 

		returns:
			klass2idx .: dict mapping classes (str) to indexes 

			word2idx .:  dict mapping words (str) to indexes 

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





	