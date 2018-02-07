'''
Created on Jan 26, 2018

@author: Varela

This module should be called from project root

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


def embed_input_lazyload(w2i_dataset_name='word2idx', embedding_dataset_name='embeddings', input_dir=TARGET_PATH):
	dataset_path= input_dir + w2i_dataset_name + '.pickle'	
	if os.path.isfile(dataset_path):
		pickle_in = open(dataset_path,"rb")
		word2idx = pickle.load(pickle_in)
		pickle_in.close()
	else:
		deep_df= propbankbr_lazyload(dataset_name='zhou')
		word2idx = df2word2idx(deep_df, col2tokenize='LEMMA')				
		embed_word2idx_persist(word2idx, dataset_name=w2i_dataset_name)		

	dataset_path= input_dir + embedding_dataset_name + '.npy'
	if os.path.isfile(dataset_path):		
		embeddings = np.load(dataset_path)		
	else:
		word2vec= embed_word2vec(dataset_name='glove_s50')		
		embeddings= idx2embedding(word2idx, word2vec)
		embed_embedding_persist(embeddings, dataset_name=embedding_dataset_name)		
		
	return word2idx, embeddings	

def embed_output_lazyload(w2i_dataset_name='klass2idx', klass_ind_dataset_name='klass_ind', input_dir=TARGET_PATH):
	dataset_path= input_dir + w2i_dataset_name + '.pickle'	
	if os.path.isfile(dataset_path):
		pickle_in = open(dataset_path,"rb")
		word2idx = pickle.load(pickle_in)
		pickle_in.close()
	else:
		deep_df= propbankbr_lazyload(dataset_name='zhou')
		word2idx = df2word2idx(deep_df, col2tokenize='ARG_Y')						
		embed_word2idx_persist(word2idx, dataset_name=w2i_dataset_name)		

	dataset_path= input_dir + klass_ind_dataset_name + '.npy'
	if os.path.isfile(dataset_path):
		klass_ind=np.load(dataset_path)
	else:
		word2vec= embed_word2vec(dataset_name='glove_s50')		
		klass_ind= idx2klass_ind(word2idx, word2vec)
		embed_embedding_persist(klass_ind, dataset_name=klass_ind_dataset_name)		
		
	return word2idx, klass_ind


def embed_word2vec(dataset_name='glove_s50'):
	embedding_path=EMBEDDING_PATH + dataset_name + '.txt'
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

	word2idx, embeddings = embed_input_lazyload() 
	print('# tokens', len(word2idx.keys()))
	print('embeddings shape', embeddings.shape)

	klass2idx, klass_ind = embed_output_lazyload() 
	print('# classes', len(klass2idx.keys()))
	print('klass_ind shape', klass_ind.shape)
	
