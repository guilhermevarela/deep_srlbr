'''
Created on Jan 26, 2018

@author: Varela

Handles categorical feature mapping to it's embedding space

Adds and embedding layer over the raw data generating
	* input 	vocabulary word2idx 
	* target  vocabulary word2idx 
	* embedding over input
	* klass_ind enconding over output

'''
#Uncomment if launched from root
# from datasets.data_propbankbr import propbankbr_lazyload
import sys
sys.path.append('datasets/')
#Uncomment if launched from /datasets
from data_propbankbr import propbankbr_lazyload
from gensim.models import KeyedVectors

import pickle
import numpy as np 

import os.path

EMBEDDING_PATH='datasets/embeddings/'
TARGET_PATH='datasets/inputs/00/'


def vocab_lazyload(column, input_dir=TARGET_PATH):
	newfeature=False
	# all columns that have been trained with an embedding model
	if column in ['FORM', 'LEMMA', 'PRED']: 
		vocab_name='word2idx'
		dataset_path= input_dir + vocab_name + '.pickle'	
		if os.path.isfile(dataset_path):
			pickle_in = open(dataset_path,"rb")
			word2idx = pickle.load(pickle_in)
			pickle_in.close()
		else:
			newfeature=True
	else:
		newfeature=True

	if newfeature:	
		deep_df= propbankbr_lazyload(dataset_name='zhou')
		word2idx = df2word2idx(deep_df, col2tokenize=column)				
		vocab_word2idx_persist(word2idx, dataset_name=column.lower() + '2idx')		

	return word2idx



def vocab_lazyload_with_embeddings(column, embedding_name='embeddings', input_dir=TARGET_PATH):
	newfeature=False
	word2idx= vocab_lazyload(column, input_dir=TARGET_PATH)
	# Standardized embeddings
	if column in ['FORM', 'LEMMA', 'PRED']: 
		dataset_path= input_dir + embedding_name + '.npy'
		if os.path.isfile(dataset_path):		
			embeddings = np.load(dataset_path)		
		else: 
			word2vec= vocab_word2vec(dataset_name='glove_s50')		
			embeddings= idx2embedding(word2idx, word2vec)
			vocab_embedding_persist(embeddings, embedding_name=embedding_name)		
	else:
		raise ValueError('{:s} doesnt support standardized embeddings'.format(column))
			
	return word2idx, embeddings	


def vocab_word2vec(dataset_name='glove_s50'):
	embedding_path=EMBEDDING_PATH + dataset_name + '.txt'
	word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")	
	return word2vec	

def vocab_embedding_persist(embeddings, dataset_name='embeddings'):
	persist_path=TARGET_PATH + dataset_name + '.npy'
	return np.save(persist_path, embeddings)

def vocab_word2idx_persist(word2idx, dataset_name='word2idx'):
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

def idx2klass_ind(word2idx, word2vec):	
	sz_vocab=max(word2idx.values())+1
	klass_ind= np.eye(sz_vocab, dtype=np.int32)	
	return klass_ind

def token2vec(token, w2v):
	try:
		vec = w2v[token]
	except KeyError:
		vec = w2v['unk']
	return vec 


if __name__== '__main__':

	word2idx, embeddings = vocab_lazyload_with_embeddings(column='LEMMA') 
	print('# tokens', len(word2idx.keys()))
	print('embeddings shape', embeddings.shape)

	klass2idx = vocab_lazyload(column='ARG_Y') 
	print('# classes', len(klass2idx.keys()))
	# print('klass_ind shape', klass_ind.shape)
	
