'''
Created on Jan 26, 2018
	@author: Varela
	
	Converts gesim w2v to Tensorflow's tensor

'''
import pandas as pd
import numpy as np 
import tensorflow as tf 


def idx2embedding(word2idx, word2vec):
	sz_embedding=len(word2vec['unk'])
	sz_vocab=max(word2idx.values())+1
	embedding= 	np.zeros((sz_vocab, sz_embedding), dtype=np.float32)
	for word, idx in word2idx.items():
		embedding[idx]=token2vec(word, word2vec)
	return embedding

def idx2onehot(word2idx, word2vec):	
	sz_vocab=max(word2idx.values())+1
	onehot= np.eye(sz_vocab, dtype=np.int32)	
	return onehot

def token2vec(token, w2v):
	try:
		vec = w2v[token]
	except KeyError:
		vec = w2v['unk']
	return vec 

# def proposition2sequence_examples(df, input_vocab=[], output_vocab=[], col_tokens=['LEMMA', 'M_R'], col_labels=['ARG']):
# 	if not(vocab):
# 		input_vocab= extract_vocab(df, column='LEMMA')

# 	if not(vocab):
# 		output_vocab= extract_vocab(df, column='ARG')		

# 	np = max(df['P_S'])	# number of propositions
# 	for p in range(1, np+1):
# 		ex= proposition2sequence_example(
# 			dataframe2dict( df[df['P_S']==p] ), 
# 			input_vocab, 
# 			output_vocab, 
# 			col_tokens=col_tokens,
# 			col_labels=col_labels
# 		)


def proposition2sequence_example(prop_dict, input_vocab, output_vocab, features=['PRED'], sequence_keys=['LEMMA', 'M_R'], target_key=['LABEL']):
	ex= tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length=len(prop_dict[target_key[0]])
	ex.context.feature['length'].int64_list.value.append(sequence_length)
	# import code; code.interact(local=dict(globals(), **locals()))			

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














