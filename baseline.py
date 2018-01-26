'''
Created on Jan 25, 2018
	@author: Varela
	
'''
import pandas as pd 
from data_transform import extract_vocab, to_embedding
from gensim.models import KeyedVectors


if __name__== '__main__':
	#Load data
	df= pd.read_csv('propbankbr/zhou.csv')
	w2v = KeyedVectors.load_word2vec_format('datasets/glove_s50.txt', unicode_errors="ignore")

	#perform data
	vocab= extract_vocab(df)
	embedding= to_embedding(vocab, w2v, 50)
	import code; code.interact(local=dict(globals(), **locals()))			
	# deep_srl= BasicLSTM() 
	# deep_srl.train(batch_sz=200, display_step=10)




