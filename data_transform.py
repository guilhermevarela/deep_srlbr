'''
Created on Jan 26, 2018
	@author: Varela
	
	Converts gesim w2v to Tensorflow's tensor

'''
import numpy as np 

def extract_vocab(df, column='LEMMA'):		
	vocab= list(df.drop_duplicates(subset=column)[column])
	vocab= map(str.lower, vocab)
	return sorted(vocab)

def to_embedding(vocab, w2v, embedding_size):
	vocab_size=len(vocab)
	embedding= 	np.zeros((vocab_size, embedding_size), dtype=np.float32)
	for i, w in enumerate(vocab):
		embedding[i]=token2vec(vocab[i], w2v)
	return embedding

def token2vec(token, w2v):
	try:
		vec = w2v[token]
	except KeyError:
		vec = w2v['unk']
	return vec 






