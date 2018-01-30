'''
Created on Jan 26, 2018
	@author: Varela
	
	Supports Tensorflow 1.4r

'''
from aux_object import *
import tensorflow as tf 

#Lazy loading prevents graph re definition
class BasicLSTM(object):
	def __init__(self, data, target, w2v):
		self.data= data 
		self.target= target
		self.w2v= w2v		
		self.initialize_batches()

		self.predict
		self.train
		self.eval

	@define_scope
	def predict(self):
		#[batch_sz, sequence_lenght, input_sz]				
		self.batch()
		outputs, states= tf.nn.dynamic_rnn(
			self.cell(), 
			self.input(), 
			sequence_lenght=self.batch_lenght
			dtype=tf.float32
		)
	@define_scope
	def train(self):		
			raise NotImplementedError		

	@define_scope
	def eval(self):	
			raise NotImplementedError

	def cell(self):
		return tf.nn.rnn_cell.BasicLSTMCell(512)

	def initialize_batches(self):
		if not(self.data):
			raise ValueError('data must be fed before batch initialization')
		
		batch_sz=200
		self.batch_sz=batch_sz
		self.input_sz=2*50+1
		self.N= len(self.data) # dictionary of examples
		self.nb= int(self.N/ batch_sz)
		self.batch=0

	def batch(self):
		# if self.batch< self.nb:
			#perform operations
			# [ (self.data[i]['LEMMA'], self.data[i]['M_R'],len(self.data[i]['LEMMA']))
			# 		for i in range(self.batch*self.batch_sz, (self.batch+1)*self.batch_sz)]
		self.batch+=1 
		if self.batch == self.nb:	
			self.batch=0
		






