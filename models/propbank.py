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



SEQUENCE_FEATURES=['IDX','ID','S','P','P_S','FORM','LEMMA','GPOS','PRED','FUNC','M_R','CTX_P-3','CTX_P-2','CTX_P-1','CTX_P+1','CTX_P+2','CTX_P+3','ARG_0','ARG_1']
SEQUENCE_FEATURES_TYPES=['int', 'int', 'int', 'int', 'int', 'txt', 'txt', 'hot', 'txt', 'txt', 'boo','txt', 'txt','txt', 'txt','txt', 'txt', 'hot', 'hot']




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
			meta= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))			
			for feat in self.features:
				if meta[feat]  == 'hot':
					keys= set(df[feat].values)
					idxs= list(range(len(keys)))
					self.onehot[feat]= OrderedDict(zip(keys, idxs))
					self.onehot_inverse= OrderedDict(zip(idxs, keys))
				# for categorical variables	
				if meta[feat] in ['hot']:
					self.data[feat] = OrderedDict(zip(
						df[feat].index, 
						[self.onehot[feat][val] 
							for val in df[feat].values]
					))
				elif meta[feat] in ['txt']:
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
		if not(filename):
			filename= '{:}{:}_{:}_{:}.pickle'.format(file_dir, self.db_name, self.lexicon_columns, self.language_model)
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
    	dump= pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    return dump 
  	
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
		meta= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))
		if meta[feature] == ['hot']:
			result= self.onehot[feature][value]
		elif meta[feature] == ['txt']:
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
		'''
		meta= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))
		result=0
		for feat in features:
			if meta[feat] in ['txt']:
				result+=len(self.embeddings['unk'])
			elif meta[feat] in ['hot']:
				result+=len(self.onehot[feat])
			else:
				result+=1
		return result

	def sequence_example(self, values, features):
		raise NotImplementedError




	


if __name__ == '__main__':
	propbank = Propbank()
	propbank.define()
	# Test()
	import code; code.interact(local=dict(globals(), **locals()))		