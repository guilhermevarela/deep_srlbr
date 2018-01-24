'''
Created on Jan 24, 2018
	@author: Varela
	
	streamlining experiments

'''
from utils import onehot_encode
from gensim.models import KeyedVectors
from gensim import corpora, models, similarities

import tensorflow as tf 
import pandas as pd 
import numpy as np 

ZHOU_HEADER=[
	'ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'PRED', 'M_R', 'LABEL'
]

class BasicLSTM(object):
	def __init__(self, hidden_sz=512, embedding='glove_s50'):
		embedding_path=  'datasets/%s.txt' % (embedding)
		#DEFINE EMBEDDINGS
		self.w2v = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")
		self.name= 'lstm_basic_%s' % (hidden_sz)

		#DEFINE TRAINING DATA
		df_train= pd.read_csv('propbankbr/zhou_devel.csv')		
		self.Xtrain= df_train[ZHOU_HEADER[:-1]]
		self.Ytrain= onehot_encode(df_train[ZHOU_HEADER[-1]].as_dataframe().as_matrix())

		#DEFINE VALIDATION DATA
		df_valid= pd.read_csv('propbankbr/zhou_valid.csv')
		self.Xvalid= df_valid[ZHOU_HEADER[:-1]]
		self.Yvalid= onehot_encode(df_valid[ZHOU_HEADER[-1]].as_dataframe().as_matrix())

		#DEFINE DATA PARAMS
		self.vocab_sz= Ytrain.shape[0]
		self.input_sz= 2*50+1 # 2 times embedding size + one
		self.n_examples_train= df_train['S_P']
		self.n_examples_valid= df_train['S_P']
		self.n_tokens= df_train['S_P']





	def load():
		raise NotImplementedError

	def train(self, lr=1e-3, epochs=1000, batch_sz=500, display_step=100):
		n_batch= int(self.n_examples / batch_sz)
		
		print('###########%s SETTINGS #############' % (self.name))
		print('learning rate\t%0.8f' % lr)
		print('epochs\t%0.0f' % epochs)
		print('n_examples\t%0.0f' % epochs)

		#DEFINE TENSORFLOW VARIABLES
		Wo= tf.Variable(tf.random_normal([n_hidden, vocab_size])) # Weights on the output layer
		bo= tf.Variable(tf.random_normal([vocab_size]))    

		#DEFINE TENSORFLOW PLACEHOLDER
		x = tf.placeholder(tf.float32, shape=(None, n_input), name='x')	
		y = tf.placeholder(tf.float32, shape=(None, vocab_size), name='y')



	

		

	def eval():
		raise NotImplementedError

	def predict():		
		raise NotImplementedError
			

	