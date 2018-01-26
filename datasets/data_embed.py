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

import numpy as np 


EMBEDDING_PATH='embeddings/'
TARGET_PATH='training/pre/00/'

def embed_word2vec(dataset_name='glove_s50'):
	embedding_path=EMBEDDING_PATH + dataset_name
	word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")	
	return word2vec	


def embed_embedding_persist(embedding, embed_name='embedding'):
	embedding_path=TARGET_PATH + dataset_name
	return np.save(embedding_path + '.npy', embedding)

def embed_word2idx_persist(embedding, embed_name='embedding'):
	embedding_path=TARGET_PATH + dataset_name
	return np.save(embedding_path + '.npy', embedding)	


def df2word2idx(df, column='LEMMA'):		
	vocab= list(df.drop_duplicates(subset=column)[column])
	vocab= list(map(str.lower, vocab))
	vocab_sz= len(vocab)
	return dict(zip(sorted(vocab),range(vocab_sz)))

if __name__== '__main__':
	deep_df= propbankbr_lazyload(dataset_name='zhou')
	word2vec = embed_word2vec(dataset_name='glove_s50')

	#perform data
	input_word2idx = df2word2idx(deep_df, column='LEMMA')
	embedding= idx2embedding(input_word2idx, word2vec)
	pickle_out= open('datasets/input_word2idx.picke','wb') 
	pickle.dump(input_word2idx, pickle_out)
	pickle_out.close()
	np.save('datasets/embedding.npy', embedding)
