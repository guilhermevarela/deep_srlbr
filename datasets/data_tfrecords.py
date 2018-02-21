'''
Created on Jan 25, 2018
	@author: Varela

	Generates tfrecords
		*devel
		*valid
		*test
	
'''
import pandas as pd 
import numpy as np 

#Uncomment if launched from root
# from datasets.data_propbankbr import  propbankbr_lazyload
# from datasets.data_embed import vocab_lazyload_with_embeddings, vocab_lazyload
#Uncomment if launched from /datasets
from data_propbankbr import  propbankbr_lazyload
from data_vocabularies import vocab_lazyload_with_embeddings, vocab_lazyload

import tensorflow as tf 

# EMBEDDING_PATH='embeddings/'
# TARGET_PATH='training/pre/00/'
# EMBEDDING_PATH='datasets/embeddings/'
# TARGET_PATH='datasets/inputs/01/'
EMBEDDING_PATH='datasets/embeddings/'
TARGET_PATH='datasets/inputs/02/'

#Must be both 1) inputs to the model 2) have a string representation
EMBEDDABLE_FEATURES=['FORM','LEMMA', 'PRED']
SEQUENCE_FEATURES=['IDX', 'P', 'ID', 'LEMMA', 'M_R', 'PRED', 'FUNC', 'ARG_0']
TARGET_FEATURE=['ARG_1']



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
		val = this_vocab[token.lower()]
	else:
		val = dict_vocabs[key][token]
	return val

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





