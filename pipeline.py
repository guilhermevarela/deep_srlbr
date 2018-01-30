'''
Created on Jan 21, 2018
	@author: Varela
	
	first approximation
'''
from gensim.models import KeyedVectors
from gensim import corpora, models, similarities

import pandas as pd 
import numpy as np 
# from scipy import shuffle
from utils import onehot_encode

import tensorflow as tf
# from tensorflow.contrib import rnn 

if __name__ == '__main__':
	w2v = KeyedVectors.load_word2vec_format('datasets/embeddings/glove_s50.txt', unicode_errors="ignore")
	# import code; code.interact(local=dict(globals(), **locals()))			

	df_train= pd.read_csv('datasets/csvs/zhou_devel.csv')
	df_valid= pd.read_csv('datasets/csvs/zhou_valid.csv')

	Xtrain= df_train.as_matrix()
	Ytrain= Xtrain[:,-1]
	Xtrain= Xtrain[:,:-1]

	Yind= onehot_encode( Ytrain	)
	batch_sz=200

	n_batch= int(Xtrain.shape[0] / batch_sz)
	epochs=100
	display_step=100
	#NLP
	vocab_size= 59
	#LSTM
	n_hidden= 512
	n_input=101
	learning_rate=1e-4

	#RNN output node weights and biases
	weights={
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
	}
	biases= {
    'out': tf.Variable(tf.random_normal([vocab_size]))    
	}

	# x = tf.placeholder(tf.float32, shape=(None, n_input,1 ), name='x')	
	x = tf.placeholder(tf.float32, shape=(None, n_input), name='x')	
	y = tf.placeholder(tf.float32, shape=(None, vocab_size), name='y')
	
	def token2vec(token):		
		try: 
			vec=w2v[token] 
		except:
			vec=w2v['unk'] 	
		return vec

	def cell():
		#1-layer LSTM with n_hidden units.
		return tf.nn.rnn_cell.BasicLSTMCell(n_hidden)


	def RNN(x, weights, biases):

		#reshape to [1, n_input]
		# x = tf.reshape(x, [-1, n_input])

		#Generate a n_input-element sequence of inputs
		#(eg. [had] [a] [general] ->  [20] [6] [33])
		x = tf.split(x, n_input, 1)


		rnn_cell=cell()
		#generate prediction 
		lstm_cell= tf.nn.rnn_cell.MultiRNNCell([rnn_cell])
		outputs, states= tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

		#there are n_input outputs but
		#we only want the last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']    

	pred = RNN(x, weights, biases)

	# Loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

	# Model evaluation
	correct_pred= tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))	

	#Initializing the variables
	init= tf.global_variables_initializer()    	

	#Launch the graph
	with tf.Session() as session:
		session.run(init)
		step=0 		
		acc_total=0 
		loss_total=0 
	
		for e in range(epochs):		
			for j in range(n_batch):
				Xbatch=Xtrain[j*batch_sz:(j+1)*batch_sz,:]
				Ybatch=  Yind[j*batch_sz:(j+1)*batch_sz,:]
				
				x_batch=np.zeros((batch_sz,n_input),dtype=np.float32)				
				for i in range(batch_sz):			

					x0= token2vec(Xbatch[i,6].lower())
					x1= token2vec(Xbatch[i,7].lower())
					x2= np.array([Xbatch[i,8]])
					x_batch[i,:]= np.concatenate((x0,x1,x2), axis=0)
					

				_, acc, loss, one_hotpred= session.run(
	    		[optimizer, accuracy, cost, pred],
	    		feed_dict={x: x_batch, y:Ybatch}            
				)    
				
				loss_total += loss 
				acc_total  += acc 
				
				if (step +1) % display_step==0:
					msg= 'Iter=' + str(step+1) 
					msg+=', Average Loss=' + "{:.6f}".format(loss_total/display_step) 
					msg+= ', Average Accuracy=' + "{:.2f}".format(100*acc_total/display_step)
					print(msg)
					acc_total=0
					loss_total=0						
				step+=1













