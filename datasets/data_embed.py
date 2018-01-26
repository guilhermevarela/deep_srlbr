'''
Created on Jan 26, 2018

@author: Varela

Adds and embedding layer over the raw data generating
	* input 	vocabulary word2idx 
	* target  vocabulary word2idx 
	* embedding over input
	* onehot enconding over output

'''
from data_propbankbr import propbankbr_lazyload

import pickle
import numpy as np 

import os.path

EMBEDDING_PATH='embeddings/'
TARGET_PATH='training/pre/00/'

# def propbankbr_lazyload(dataset_name='zhou'):
# 	dataset_path= TARGET_PATH + '/{}.csv'.format(dataset_name)
# 	if os.path.isfile(dataset_path):
# 		df= pd.read_csv(dataset_path)		
# 	else:
# 		df= propbankbr_parser2()
# 		propbankbr_persist(df, split=True, dataset_name=dataset_name)		  
# 	return df 

def embed_input_lazyload(w2i_dataset_name='input_word2idx', embedding_dataset_name='embedding'):
	dataset_path= TARGET_PATH + w2i_dataset_name + '.pickle'
	if os.path.isfile(dataset_path):
		pickle_in = open(dataset_path,"rb")
		word2idx = pickle.load(pickle_in)
		pickle_in.close()
	else:
		deep_df= propbankbr_lazyload(dataset_name='zhou')
		word2idx = df2word2idx(deep_df, col2tokenize='LEMMA')				
		embed_word2idx_persist(word2idx, dataset_name=w2i_dataset_name)		

	dataset_path= TARGET_PATH + embedding_dataset_name + '.npy'
	if os.path.isfile(dataset_path):		
		embeddings = np.load(dataset_path)		
	else:
		word2vec= embed_word2vec(dataset_name='glove_s50')		
		embeddings= idx2embedding(word2idx, word2vec)
		embed_embedding_persist(embeddings, dataset_name=embedding_dataset_name)		
		
	return word2idx, embeddings	

def embed_output_lazyload(w2i_dataset_name='output_word2idx', onehot_dataset_name='onehot'):
	dataset_path= TARGET_PATH + w2i_dataset_name + '.pickle'
	if os.path.isfile(dataset_path):
		pickle_in = open(dataset_path,"rb")
		word2idx = pickle.load(pickle_in)
		pickle_in.close()
	else:
		deep_df= propbankbr_lazyload(dataset_name='zhou')
		word2idx = df2word2idx(deep_df, col2tokenize='LABEL')				
		embed_word2idx_persist(word2idx, dataset_name=w2i_dataset_name)		

	dataset_path= TARGET_PATH + onehot_dataset_name + '.npy'
	if os.path.isfile(dataset_path):
		onehot=np.load(dataset_path)
	else:
		word2vec= embed_word2vec(dataset_name='glove_s50')		
		onehot= idx2onehot(word2idx, word2vec)
		embed_embedding_persist(onehot, dataset_name=onehot_dataset_name)		
		
	return word2idx, onehot


def embed_word2vec(dataset_name='glove_s50'):
	embedding_path=EMBEDDING_PATH + dataset_name
	word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")	
	return word2vec	

def embed_embedding_persist(embeddings, dataset_name='embeddings'):
	persist_path=TARGET_PATH + dataset_name + '.npy'
	return np.save(persist_path, embeddings)

def embed_word2idx_persist(word2idx, dataset_name='word2idx'):
	persist_path=TARGET_PATH + dataset_name + '.pickle'
	pickle_out= open(persist_path,'wb') 
	pickle.dump(word2idx, pickle_out)
	pickle_out.close()
	return word2idx

def df2word2idx(df, col2tokenize='LEMMA'):	
	vocab= list(df.drop_duplicates(subset=col2tokenize)[col2tokenize])
	vocab= list(map(str.lower, vocab))
	vocab_sz= len(vocab)
	return dict(zip(sorted(vocab),range(vocab_sz)))

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


if __name__== '__main__':
	# deep_df= propbankbr_lazyload(dataset_name='zhou')
	# word2vec = embed_word2vec(dataset_name='glove_s50')

	#perform data
	# word2idx = df2word2idx(deep_df, col2tokenize='LEMMA')
	# embedding= idx2embedding(word2idx, word2vec)
	# embed_word2idx_persist(word2idx)
	# embed_embedding_persist(embedding)
	input_word2idx, embeddings = embed_input_lazyload() 
	print('# tokens', len(input_word2idx.keys()))
	print('embeddings shape', embeddings.shape)

	output_word2idx, onehot = embed_output_lazyload() 
	print('# classes', len(output_word2idx.keys()))
	print('onehot shape', onehot.shape)
	
