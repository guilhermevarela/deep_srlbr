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
import copy
import os.path
import re
import string

EMBEDDING_PATH='datasets/embeddings/' # use scripts
# EMBEDDING_PATH='../datasets/embeddings/' # use notebooks
TARGET_PATH='datasets/inputs/01/'


def vocab_lazyload(column, input_dir=TARGET_PATH):
	# all columns that have been trained with an embedding model
	if column in ['FORM', 'LEMMA', 'PRED']: 
		dict_name='word2idx'
		lower_case=True
	else:
		dict_name=column.lower() + '2idx'
		lower_case=False

	dataset_path= input_dir + dict_name + '.pickle'	
	#Check if exists	
	if os.path.isfile(dataset_path):
		pickle_in = open(dataset_path,"rb")
		word2idx = pickle.load(pickle_in)
		pickle_in.close()
	else:
		deep_df= propbankbr_lazyload(dataset_name='zhou')		
		word2idx = df2word2idx(deep_df, col2tokenize=column,lower_case=lower_case)				

		vocab_word2idx_persist(word2idx, dict_name=dict_name)		

	return word2idx



def vocab_lazyload_with_embeddings(column, embeddings_id='embeddings', input_dir=TARGET_PATH, verbose=False):
	not_found_count=0
	word2idx= vocab_lazyload(column, input_dir=input_dir)
	updated_word2idx= copy.deepcopy(word2idx)

	#Raw embeddings model and post processed npy matrix
	embeddings_txtfile= 'glove_s50' if embeddings_id == 'embeddings' else embeddings_id
	embeddings_npyfile= embeddings_id
	# Standardized embeddings
	if column in ['FORM', 'LEMMA', 'PRED', 'FUNC', 'CTX_P-3', 'CTX_P-2', 'CTX_P-1', 'CTX_P+1', 'CTX_P+2', 'CTX_P+3']: 
		dataset_path= input_dir + embeddings_npyfile + '.npy'
		lower_case=True	
		if os.path.isfile(dataset_path):		
			embeddings = np.load(dataset_path)		
		else: 
			word2vec= vocab_word2vec(dataset_name=embeddings_txtfile)		
			embeddings, updated_word2idx= idx2embedding(word2idx, word2vec, verbose=verbose)			
			vocab_embedding_persist(embeddings, dataset_name=embeddings_npyfile)		
			if not(max(updated_word2idx.values())==max(word2idx.values())):
				vocab_word2idx_persist(updated_word2idx, dict_name='word2idx')					
	else:		
		lower_case=False
		raise ValueError('{:} doesnt support standardized embeddings'.format(column))
			
	return updated_word2idx, embeddings	


def vocab_word2vec(dataset_name='glove_s50'):
	embedding_path=EMBEDDING_PATH + dataset_name + '.txt'
	word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")	
	return word2vec	

def vocab_embedding_persist(embeddings, dataset_name='embeddings'):
	persist_path=TARGET_PATH + dataset_name + '.npy'
	return np.save(persist_path, embeddings)

def vocab_word2idx_persist(word2idx, dict_name='word2idx'):
	persist_path=TARGET_PATH + dict_name + '.pickle'
	pickle_out= open(persist_path,'wb') 
	pickle.dump(word2idx, pickle_out)
	pickle_out.close()
	return word2idx

def df2word2idx(df, col2tokenize='LEMMA', lower_case=True):		
	vocab= list(df.drop_duplicates(subset=col2tokenize)[col2tokenize])
	if lower_case:
		vocab= list(map(str.lower, vocab))
	vocab_sz= len(vocab)
	return dict(zip(sorted(vocab),range(vocab_sz)))

def idx2embedding(word2idx, word2vec, verbose=False):	
	not_found_words=[]
	word2idx = copy.deepcopy(word2idx)

	for word in word2idx:
		token=vocab_preprocess(word)
		_, found=token2vec(token, word2vec,verbose=verbose)
		if not(found):
			not_found_words.append(word)
	
	sz_embedding=len(word2vec['unk'])
	sz_vocab=len(word2idx)-len(not_found_words)+1
	embedding= 	np.zeros((sz_vocab, sz_embedding), dtype=np.float32)
	j=0
	for i, word in enumerate(word2idx):	
		if i==0:	
			embedding[i]=word2vec['unk']
		elif word in not_found_words:
			word2idx[word]=0
			j+=1 
		else:			
			token=vocab_preprocess(word)
			embedding[i-j], _ =token2vec(word, word2vec)
			word2idx[word]=i-j

	print('not found words', len(not_found_words))
	return embedding, word2idx	

def token2vec(token, w2v, verbose=False):
	found=True
	try:
		vec = w2v[token]
	except KeyError:
		found=False
		if verbose:
			print('token {} not found'.format(token))
		vec = w2v['unk']
	return vec, found 

def vocab_preprocess(text):
	'''
		vocab_preprocess text according to embedding
		args:

		returns:
			token .: process string before looking up
		ref https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py
			/notebooks/corpus-word-preprocessing
	'''
	token=text.lower()
	re_punctuation= re.compile(r'[{:}]'.format(string.punctuation), re.UNICODE)
	re_number= re.compile(r'^\d+$')
	token=re_punctuation.sub('', token)
	token=re_number.sub('0', token)

	return token 

if __name__== '__main__':
	# embedding_path=EMBEDDING_PATH  'glove_s50.txt'
	# word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")	
	# import code; code.interact(local=dict(globals(), **locals()));
	word2idx, embeddings = vocab_lazyload_with_embeddings(column='LEMMA', embeddings_id='wang2vec_s100', verbose=True) 
	print('# tokens', len(word2idx.keys()))
	print('embeddings shape ', embeddings.shape)
	func2idx = vocab_lazyload('FUNC') 
	print('# feature FUNC', len(func2idx.keys()))
	arg02idx = vocab_lazyload('ARG_0') 
	print('# classes ARG_0', len(arg02idx.keys()))
	arg12idx = vocab_lazyload('ARG_1') 
	print('# classes ARG_1', len(arg12idx.keys()))
	
	
