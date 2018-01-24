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
		self.hidden_sz=hidden_sz
		self.vocab_sz= Ytrain.shape[0]
		self.input_sz= 2*50+1 # 2 times embedding size + one
		self.n_examples_train= max(df_train['S_P'])
		self.n_examples_valid= max(df_valid['S_P'])
		self.n_tokens= np.unique(df_train['S_P'].as_dataframe().as_matrix())
		self.n_tuples= df_train.shape[0]

		#Initialize cell
		self.rnn_cell= rnn.core_rnn_cell.BasicLSTMCell(hidden_sz)



	def load():
		raise NotImplementedError

	def train(self, lr=1e-3, epochs=1000, batch_sz=500, display_step=100):
		#DEFINE EXECUTION VARIABLES
		n_batch= int(self.n_examples_train / batch_sz)
		
		header='###########%s SETTINGS #############' % (self.name)
		print(header)
		print('learning rate\t%0.8f' % lr)
		print('epochs\t%0.0f' % epochs)
		print('n examples train\t%0.0f' % epochs)
		print('n examples valid\t%0.0f' % epochs)
		print('n examples train\t%0.0f' % epochs)
		print('n examples valid\t%0.0f' % epochs)
		print('batch size\t%0.0f' % epochs)
		print('n batches\t%0.0f' % epochs)
		print('#'*len(header))

		#DEFINE TENSORFLOW VARIABLES
		Wo= tf.Variable(tf.random_normal([self.hidden_sz, self.vocab_sz])) # Weights on the output layer
		bo= tf.Variable(tf.random_normal([self.vocab_sz]))    

		#DEFINE TENSORFLOW PLACEHOLDER
		x = tf.placeholder(tf.float32, shape=(None, self.input_sz), name='x')	
		y = tf.placeholder(tf.float32, shape=(None, self.vocab_sz), name='y')

		predict_op= self.forward_op(x, Wo, bo)
		cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_op, labels=y))
		train_op= tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
		
		# Model evaluation
		eval_op=   tf.equal(tf.argmax(predict_op,1), tf.argmax(y,1))
		accuracy_op= tf.reduce_mean(tf.cast(correct_pred, tf.float32))		

		

	def eval():
		raise NotImplementedError

	def predict():		
		raise NotImplementedError

	def forward_op(x, Wo, bo):
		x = tf.split(x, n_input, 1)
		
		#generate prediction 
		lstm_cell= rnn.MultiRNNCell([self.rnn_cell])
		outputs, states= rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		#there are n_input outputs but
		#we only want the last output
		return tf.matmul(outputs[-1], Wo) + bo
			

	