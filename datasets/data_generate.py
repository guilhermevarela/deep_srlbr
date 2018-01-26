'''
Created on Jan 25, 2018
	@author: Varela
	
'''
import pandas as pd 
import numpy as np 

from data_transform import *
from gensim.models import KeyedVectors

import tensorflow as tf 




if __name__== '__main__':
	#Load data
	deep_df= pd.read_csv('propbankbr/zhou.csv')
	word2vec = KeyedVectors.load_word2vec_format('embeddings/glove_s50.txt', unicode_errors="ignore")

	#perform data
	input_word2idx = df2word2idx(deep_df, column='LEMMA')
	embedding= idx2embedding(input_word2idx, word2vec)
	pickle_out= open('datasets/input_word2idx.picke','wb') 
	pickle.dump(input_word2idx, pickle_out)
	pickle_out.close()
	np.save('datasets/embedding.npy', embedding)
	

	target_word2idx = df2word2idx(deep_df, column='LABEL')
	onethot= idx2embedding(target_word2idx, word2vec)
	pickle_out=open('datasets/target_word2idx.picke','wb')
	pickle.dump(target_word2idx, pickle_out)
	pickle_out.close()

	np.save('datasets/onehot.npy', onethot)
	
	deep_df= pd.read_csv('propbankbr/zhou_devel.csv')
	trecords_path='datasets/devel.tfrecords'

	np = max(deep_df['P_S'])	# number of propositions

	with open(trecords_path) as f:
		writer= tf.python_io.TFRecordWriter(f.name)

		for p in range(1, np+1):
			ex= proposition2sequence_example(
				df2data_dict( deep_df[deep_df['P_S']==p] ), 
				input_word2idx, 
				target_word2idx 
			)	

			writer.write(ex.SerializeToString())        
    
		writer.close()
		print('Wrote to {}'.format(f.name))		





