'''
Created on Jan 24, 2018
	@author: Varela
	
	streamlining experiments

'''
from utils import onehot_encode, shuffle_by_proposition, shuffle_by_rows, token2vec
from gensim.models import KeyedVectors
from gensim import corpora, models, similarities


import pandas as pd 
import numpy as np 

import tensorflow as tf 


ZHOU_HEADER=[
	'ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'PRED', 'M_R', 'LABEL'
]

class BasicLSTM(object):
	def __init__(self, hidden_sz=512, embedding='glove_s50'):
		embedding_path=  'datasets/%s.txt' % (embedding)

		#DEFINE EMBEDDINGS
		self.w2v = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")
		self.name= 'lstm_basic_%s' % (hidden_sz)
		self.input_sz= 2*50+1 # 2 times embedding size + one

		#DEFINE TRAINING DATA
		df_train= pd.read_csv('propbankbr/zhou_devel.csv')		
		self.df_train= df_train[ZHOU_HEADER[:-1]]
		# self.Ytrain= onehot_encode(df_train[ZHOU_HEADER[-1]].to_frame().as_matrix())
		# import code; code.interact(local=dict(globals(), **locals()))			

		#DEFINE VALIDATION DATA
		df_valid= pd.read_csv('propbankbr/zhou_valid.csv')
		self.Xvalid= self._embedding_lookup(df_valid[ZHOU_HEADER[:-1]].as_matrix())
		# self.Yvalid= onehot_encode(df_valid[ZHOU_HEADER[-1]].to_frame().as_matrix())
		self.Ytrain, self.Yvalid= self._output_preprocess(df_train, df_valid)	

		#DEFINE DATA PARAMS
		self.hidden_sz=hidden_sz
		self.vocab_sz= self.Ytrain.shape[1]		
		self.n_examples_train= max(df_train['P_S'])
		self.n_examples_valid= max(df_valid['P_S']) - self.n_examples_train
		self.n_tokens= len(np.unique(df_train['LEMMA'].to_frame().as_matrix()))
		self.n_tuples= df_train.shape[0]

		#Initialize cell
		self.B_LSTM= tf.nn.rnn_cell.BasicLSTMCell(hidden_sz)



	def load(self):
		raise NotImplementedError

	def train(self, lr=1e-3, epochs=100, batch_sz=500, display_step=10):
		#DEFINE EXECUTION VARIABLES
		n_batch= int(self.n_examples_train / batch_sz)
		
		header='###########%s SETTINGS #############' % (self.name)
		print(header)
		print('learning rate\t%0.8f' % lr)
		print('epochs\t%0.0f' % epochs)
		print('n examples train\t%d' % self.n_examples_train)
		print('n examples valid\t%d' % self.n_examples_valid)
		print('n tokens\t%d' % self.n_tokens)
		print('n tuples\t%d' % self.n_tuples)
		print('batch size\t%0.0f' % batch_sz)
		print('n batches\t%0.0f' % n_batch)
		print('#'*len(header))

		#DEFINE TENSORFLOW VARIABLES
		Wo= tf.Variable(tf.random_normal([self.hidden_sz, self.vocab_sz])) # Weights on the output layer
		bo= tf.Variable(tf.random_normal([self.vocab_sz]))    

		#DEFINE TENSORFLOW PLACEHOLDER
		x = tf.placeholder(tf.float32, shape=(None, self.input_sz), name='x')	
		y = tf.placeholder(tf.float32, shape=(None, self.vocab_sz), name='y')

		predict_op= self.forward_op(x, Wo, bo)
		cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_op, labels=y))
		train_op= tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost_op)
		
		# Model evaluation
		eval_op=   tf.equal(tf.argmax(predict_op,1), tf.argmax(y,1))
		accuracy_op= tf.reduce_mean(tf.cast(eval_op, tf.float32))		

		#Initialization
		init= tf.global_variables_initializer()    	
		with tf.Session() as session:
			session.run(init)
			step=0
			loss_total=0
			acc_total=0
			for e in range(epochs):
				#shuffle
				idx, idxb= shuffle_by_proposition(self.df_train, batch_sz=batch_sz)				
				Xtrain=self.df_train.reindex(idx, copy=True).as_matrix()
				Ytrain=shuffle_by_rows(self.Ytrain, idx)
				for j in range(n_batch):
					Xbatch=self._embedding_lookup(Xtrain[idxb[j]:idxb[j+1],:])
					Ybatch=Ytrain[idxb[j]:idxb[j+1],:]
					nb= Xbatch.shape[0]

					_, acc, loss, one_hotpred= session.run(
	    			[train_op, accuracy_op, cost_op, predict_op],
	    			feed_dict={x: Xbatch, y:Ybatch}            
					)    
					
				loss_total += loss 
				acc_total  += acc 
				
				if (step +1) % display_step==0:
					valid_acc, valid_loss = session.run(
						[accuracy_op, cost_op],
	    			feed_dict={x: self.Xvalid, y: self.Yvalid}            
					)
					msg= 'Iter=' + str(step+1) 
					msg+= ', Average Insample-Loss=' + "{:.6f}".format(loss_total/display_step) 
					msg+= ', Average Insample-Accuracy=' + "{:.2f}".format(100*acc_total/display_step)
					msg+= ',  Outsample-Loss=' + "{:.6f}".format(valid_loss) 
					msg+= ',  Outsample-Accuracy=' + "{:.2f}".format(100*valid_acc)
					print(msg)
					acc_total=0
					loss_total=0						
				step+=1	
		

	def eval():
		raise NotImplementedError

	def predict():		
		raise NotImplementedError

	def forward_op(self, x, Wo, bo):
		x = tf.split(x, self.input_sz, 1)
		
		#generate prediction 
		lstm_cell= tf.nn.rnn_cell.MultiRNNCell([self.B_LSTM])
		outputs, states= tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

		#there are n_input outputs but
		#we only want the last output
		return tf.matmul(outputs[-1], Wo) + bo
			
	def _embedding_lookup(self, X):		
		N= X.shape[0] 
		Xout= np.zeros((N, self.input_sz), dtype=np.float32)
		for n in range(N):
			Xout[n,:] =np.concatenate(
				( token2vec(X[n, 5], self.w2v),
					token2vec(X[n, 6], self.w2v),
					np.array([X[n, 7]])
				), axis=0
			)
		return Xout

	def _output_preprocess(self, df_train, df_valid):	
		df = pd.concat((df_train, df_valid), axis=0)
		Y  = onehot_encode(df[ZHOU_HEADER[-1]].to_frame().as_matrix())
		
		Ytrain, Yvalid= Y[:df_train.shape[0],:],Y[-df_valid.shape[0]:,:]
		return Ytrain, Yvalid


	