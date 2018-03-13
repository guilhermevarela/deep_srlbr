'''
Created on Mar 08, 2018
	@author: Varela
	
	ref:
		https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py

	preprocessing rules:	
		Script used for cleaning corpus in order to train word embeddings.
		All emails are mapped to a EMAIL token.
		All numbers are mapped to 0 token.
		All urls are mapped to URL token.
		Different quotes are standardized.
		Different hiphen are standardized.
		HTML strings are removed.
		All text between brackets are removed.
		All sentences shorter than 5 tokens were removed.

'''
import sys
sys.path.append('datasets/')

import numpy as np 
import pandas as pd
import pickle


from utils import fetch_word2vec, fetch_corpus_exceptions, preprocess
from collections import OrderedDict

CSVS_DIR= './datasets/csvs/'



SEQUENCE_FEATURES=      ['IDX',  'ID',   'S',   'P', 'P_S', 'LEMMA','GPOS','PRED','FUNC', 'M_R','CTX_P-3','CTX_P-2','CTX_P-1','CTX_P+1','CTX_P+2','CTX_P+3','ARG_0','ARG_1']
SEQUENCE_FEATURES_TYPES=['int', 'int', 'int', 'int', 'int',   'txt', 'hot', 'txt', 'txt', 'int',    'txt',    'txt',    'txt',    'txt',    'txt',    'txt',  'hot',  'hot']
META= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))

DATASET_SIZE= 5931
DATASET_TRAIN_SIZE= 5099
DATASET_VALID_SIZE= 569
DATASET_TEST_SIZE=  263



class _PropIter(object):
	'''
	Translates propositions 

	'''	
	def __init__(self, low, high, fnc):
		self.current= low 
		self.high= high
		self.fnc= fnc

	def __iter__(self):
		return self

	def __next__(self): 
		if self.current > self.high:
			raise StopIteration
		else:
			self.current += 1
			return self.current - 1, self.fnc(self.current-1)



class Propbank(object):
	'''
	Translates propositions to features providing both the 
		data itself as meta data to machine learning experiments

	'''	
	
	def __init__(self):
		'''
			Returns new instance of propbank class - instantiates the data structures
			
			returns:
				propbank		.: object an instance of the Propbank class
		'''
		self.lexicon=set([])
		self.lex2tok={}
		self.tok2idx={}
		self.idx2tok={}
		self.embeddings=np.array([])

		self.onehot={}
		self.onehot_inverse={}
		self.data={}

	
	def total_words(self):
		return len(self.lexicon)	

	def define(self, 
		db_name='zhou_1', lexicon_columns=['LEMMA'], language_model='glove_s50', verbose=True):

		if (self.total_words() == 0):
			path=  '{:}{:}.csv'.format(CSVS_DIR, db_name)
			df= pd.read_csv(path)

			self.db_name= db_name
			self.lexicon_columns= lexicon_columns
			self.language_model= language_model


			for col in lexicon_columns:
				self.lexicon= self.lexicon.union(set(df[col].values))

			word2vec= fetch_word2vec(self.language_model)
			embeddings_sz = len(word2vec['unk'])
			vocab_sz= self.total_words()

			
			# Preprocess
			if verbose:			
				print('Processing total lexicon is {:}'.format(self.total_words())) 
					
			self.lex2tok= preprocess(list(self.lexicon), word2vec)
			self.tok2idx= {'unk':0}			
			self.idx2tok= {0:'unk'}			
			self.embeddings= np.zeros((vocab_sz, embeddings_sz),dtype=np.float32)
			self.embeddings[0]= word2vec['unk']

			tokens= set(self.lex2tok.values())
			idx=1
			for token in list(tokens):
				if not(token in self.tok2idx):
					self.tok2idx[token]=idx
					self.idx2tok[idx]=token
					self.embeddings[idx]= word2vec[token]
					idx+=1
			
					
			self.features= list(df.columns)

			for feat in self.features:
				if feat in META:
					if META[feat]  == 'hot':
						keys= set(df[feat].values)
						idxs= list(range(len(keys)))
						self.onehot[feat]= OrderedDict(zip(keys, idxs))
						self.onehot_inverse= OrderedDict(zip(idxs, keys))
					# for categorical variables	
					if META[feat] in ['hot']:
						self.data[feat] = OrderedDict(zip(
							df[feat].index, 
							[self.onehot[feat][val] 
								for val in df[feat].values]
						))
					elif META[feat] in ['txt']:
						self.data[feat] = OrderedDict(zip(
							df[feat].index, 
							[self.tok2idx[self.lex2tok[val]] 
								for val in df[feat].values]
						))
					else:
						self.data[feat] = OrderedDict(df[feat].to_dict())
		else:
			raise	Exception('Lexicon and embeddings already defined')		

	def persist(self, file_dir, filename=''):
		'''
  		Serializes this object in pickle format @ file_dir+filename
  		args:
  			file_dir	.: string representing the directory
  		 	filename  .: string (optional) filename
  	'''
		if not(filename):
			strcols= '_'.join( [col for col in self.lexicon_columns])
			filename= '{:}{:}_{:}_{:}.pickle'.format(file_dir, self.db_name, strcols, self.language_model)
		else:
			filename= '{:}{:}.pickle'.format(file_dir, filename)		

		with open(filename, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

	def recover(self, file_path):		
		'''
		Returns a copy from the instanced saved @file_path
		args:
			file_path .: string a full path to a serialized dump of propbank object

		returns:
			 object   .: propbank object instanced saved at file_path
		'''
		with open(file_path, 'rb') as f:
			copy= pickle.load(f)		
		return copy
  	
	def encode(self, value, feature):
		'''
			Returns feature representation
			args:
				value   		.: string token before embeddings or encoding strategies

				feature 	.: string feature name

			returns:				
				result    .: list<> with the numeric representation of word under feature
		'''
		result= value 
		
		if META[feature] == ['hot']:
			result= self.onehot[feature][value]
		elif META[feature] == ['txt']:
			result= self.tok2idx[self.lex2tok[value]]
		return result

	def decode(self, idx, feature):
		'''
			Returns feature word / int representation
			args:
				idx 			.: int   representing a token
				feature	  .: sring representing the feature name
				
			returns:
				result    .: string or int representing word/ value of feature
		'''				
		raise NotImplementedError

	def size(self, features):	
		'''
			Returns the dimension of the features 
			args:
				features : list<str>  with the features names

			returns:
				sz 			.: int indicating the total feature size
		'''
		
		sz=0
		for feat in features:
			if META[feat] in ['txt']:
				sz+=len(self.embeddings['unk'])
			elif META[feat] in ['hot']:
				sz+=len(self.onehot[feat])
			else:
				sz+=1
		return sz


	def sequence_example_int32(self, idx, features):
		'''
			Produces a sequence from values 

			args:
				idx   	 .: 
				features .: list<str>  with the features names

			returns:
				 				 .: int32<features> indicating int value of features
		'''		
		return [self.data[feat][idx] for feat in features]
		

	def sequence_example_raw(self, rawvalues, features):
		'''
			Produces a sequence from values 

			args:
				values   : list<str>  rawtext from csv
				features : list<str>  with the features names

			returns:
				sz 			.: float32<size> indicating values converted to example
		'''
		feat2sz= { feat: self.size(feat)
			for feat in features}		

		sz=sum(feat2sz.values())  	
		sequence= np.zeros((1,sz),dtype=np.float32)

		i=0
		j=0
		for feat, sz in feat2sz.items():
			sequence[j:j+sz]= float(self.encode( rawvalues[i], feat))
			i+=1
			j=sz 

		return sequence		

	def feature(self, feature):
		'''
			Returns a feature from data

			args:
				feature : string representing a feature from data

			returns:
				index  .: int<T> indicating values converted to example
				values .: int<T> as index
		
		'''
		#unzips features
		index, values=  zip(*self.data[feature].items())
		return index, values



	def itertrain(self, features):
		fn = lambda x : self.sequence_example_int32(x, features)

		high=-1
		low=-1		
		for idx, val in list(zip(*self.feature('P'))):
			if val > low and low==-1:
				low= idx 
			
			if val > DATASET_TRAIN_SIZE and high==-1:				
				high= idx-1  
				break
		
		return _PropIter(low, high, fn)

	def itervalid(self, features):
		fn = lambda x : self.sequence_example_int32(x, features)

		high=-1
		low=-1
		for idx, val in list(zip(*self.feature('P'))):
			if val > DATASET_TRAIN_SIZE and low==-1:
				low= idx 
			
			if val > DATASET_TRAIN_SIZE + DATASET_VALID_SIZE and high==-1:				
				high= idx-1  			
				break
		
		return _PropIter(low, high, fn)		




	


if __name__ == '__main__':
	propbank = Propbank().recover('zhou_1_LEMMA_glove_s50.pickle')
	# propbank.define()
	# propbank.persist('')
	
	min_idx=99999999999
	min_p=99999999999
	min_s=99999999999
	for idx, values in propbank.itervalid(['S','P','GPOS']):
		if min_idx> idx:
			min_idx=idx
		if min_p > values[0]:
			min_p=values[0] 

		if min_s > values[1]:
			min_s=values[1]

	print(min_idx,min_p, min_s)
	# Test()
	import code; code.interact(local=dict(globals(), **locals()))		