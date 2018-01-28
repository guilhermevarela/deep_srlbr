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

from data_propbankbr import  propbankbr_lazyload
from data_embed import embed_input_lazyload, embed_output_lazyload

import tensorflow as tf 

EMBEDDING_PATH='embeddings/'
TARGET_PATH='training/pre/00/'


def proposition2sequence_example(prop_dict, input_vocab, output_vocab, features=['PRED'], sequence_keys=['LEMMA', 'M_R'], target_key=['LABEL']):
	ex= tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length=len(prop_dict[target_key[0]])
	ex.context.feature['length'].int64_list.value.append(sequence_length)

	for feature_name in features:
		token= prop_dict[feature_name][0]
		ex.context.feature[feature_name].int64_list.value.append( input_vocab[token] )

	#Make a dictionary of feature_lists
	sequence_dict={}
	for key in sequence_keys:
		sequence_dict[key]= ex.feature_lists.feature_list[key]
		for token in prop_dict[key]:					
			if isinstance(token, str):
				sequence_dict[key].feature.add().int64_list.value.append(input_vocab[str(token).lower()])
			else:
				sequence_dict[key].feature.add().int64_list.value.append(token)

	f1_targets= ex.feature_lists.feature_list['targets']
	for key in target_key:
		for token in prop_dict[key]:					
			f1_targets.feature.add().int64_list.value.append(output_vocab[str(token).lower()])
	
	return ex	
	
def df2data_dict(df):	
	return df.to_dict(orient='list')

def data_dict2word2idx(data_dict, key='LEMMA'):		
	vocab= list(set(data_dict[key]))
	vocab= map(str.lower, vocab)
	vocab_sz= len(vocab)
	return dict(zip(sorted(vocab),range(vocab_sz)))



if __name__== '__main__':
	tfrecords_path= TARGET_PATH + 'devel2.tfrecords'

	df=propbankbr_lazyload('zhou_devel')
	np = max(df['P_S'])	# number of propositions

	word2idx, _ =embed_input_lazyload()
	klass2idx, _ =embed_output_lazyload()
	with open(tfrecords_path, 'w+') as f:
		writer= tf.python_io.TFRecordWriter(f.name)

		for p in range(1, np+1):
			df_prop= df[df['P_S']==p]
			ex= proposition2sequence_example(
				df2data_dict( df_prop ), 
				word2idx, 
				klass2idx 
			)	
			writer.write(ex.SerializeToString())        
    
		writer.close()
		print('Wrote to {}'.format(f.name))		





