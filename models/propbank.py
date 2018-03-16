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
ROOT_DIR = '/'.join(sys.path[0].split('/')[:-1]) #UGLY PROBLEM FIXES TO LOCAL ROOT --> import config
sys.path.append(ROOT_DIR)
sys.path.append('datasets/')

import config as conf 
import numpy as np 
import pandas as pd
import pickle


from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess
from collections import OrderedDict

CSVS_DIR= './datasets/csvs/'


class _PropbankIterator(object):
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
		data itself as conf.META data to machine learning experiments

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
		self.hotone={}
		self.data={}

	
	@classmethod		
	def recover(cls, file_path):		
		'''
		Returns a copy from the instanced saved @file_path
		args:
			file_path .: string a full path to a serialized dump of propbank object

		returns:
			 object   .: propbank object instanced saved at file_path
		'''
		with open(file_path, 'rb') as f:
			propbank_instance= pickle.load(f)		
		return propbank_instance

	def total_words(self):
		return len(self.lexicon)	

	def define(self, 
		db_name='db_pt', lexicon_columns=['LEMMA'], language_model='glove_s50', verbose=True):

		if (self.total_words() == 0):
			path=  '{:}{:}.csv'.format(CSVS_DIR, db_name, index_col=1)
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
			
					
			self.features= set(df.columns).intersection(set(conf.SEQUENCE_FEATURES))
			self.features= self.features.union(set(['INDEX']))
			
			#ORDER of features will be alphabetic this is important when 
			#features become tensors
			for feat in self.features:
				#ALWAYS ADD INDEX in order to make merge possible after training
				if feat in ['INDEX']: 
					self.data[feat]  = OrderedDict(zip(df.index, df.index))
				else: 
					if feat in conf.META:
						if conf.META[feat]  == 'hot':
							keys= set(df[feat].values)
							idxs= list(range(len(keys)))
							self.onehot[feat]= OrderedDict(zip(keys, idxs))
							self.hotone[feat]= OrderedDict(zip(idxs, keys))
						# for categorical variables	
						if conf.META[feat] in ['hot']:
							self.data[feat] = OrderedDict(zip(
								df[feat].index, 
								[self.onehot[feat][val] 
									for val in df[feat].values]
							))
						elif conf.META[feat] in ['txt']:
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

  	
	def decode(self, value, feature, apply_embeddings=False):
		'''
			Returns feature word from indexed representation
				this process may have losses since some on the lexicon are mapped to the same lemma
				(That applies to column LEMMA)

			
			args:
				value 			.: int   representing a db record index
				feature	    .: sring representing db column name
				
			returns:
				raw       .: string or int representing the value within original database
		'''
		result= value 		
		if conf.META[feature] in ['hot']:
			result= self.hotone[feature][value]

			if apply_embeddings:
				sz= self.size(feature)
				tmp= np.zeros((sz,),dtype=np.int32)
				tmp[result]=1
				result=tmp 

		elif conf.META[feature] in ['txt']:
			result= self.idx2tok[value]

			if apply_embeddings:
				result= self.embeddings[result]
		return result


	def size(self, features):	
		'''
			Returns the dimension of the features 
			args:
				features : str or list<str>  feature name or list of feature names

			returns:
				sz 			.: int indicating the total feature size
		'''
		if isinstance(features,str):
			sz= self._feature_size(features)
		else:
			sz=0
			for feat in features:
				sz+=self._feature_size(feat)
		return sz

	def sequence_obsexample(self, idx, features, as_dict=False, decode=False):
		'''
			Produces a record or observation from a sequence

			args:
				idx 				.: int   representing a db record index
				features	  .: list<sring> representing db column name
				as_dict	    .: bool  if 1 then return as dictionary, else list
				decode	    .: bool  if 1 then turn to lemma of string, else leave as index

			returns:
				raw 				.: list<features> or dict<features> indicating int value of features
		'''		
		if as_dict:
			if decode:
				return {feat: self.decode(self.data[feat][idx], feat)
					for feat in features}
			else:
				return {feat: self.data[feat][idx] for feat in features}
		else:
			if decode:
				return [self.decode(self.data[feat][idx], feat)
					for feat in features]
			else:
				return [self.data[feat][idx] for feat in features]
		

	def sequence_obsexample2(self, categories, features, embeddify=False):
		'''
			Produces a sequence from categorical values 
				(an entry) from the dataset

			args:
				values   .: list<str>  rawtext from csv
				features .: list<str>  with the features names
				embeddify   .: bool if false will return categorical indexes
												 if  true   will apply embeddings or onehot

			returns:
				example 	.: numerical<N> if embeddify=True then N = self.size(features) 
																	else N=len(categories)	
		'''		
		
		feat2sz= { feat: self.size(feat)
			for feat in features}		

		sz=sum(feat2sz.values())  	
		if embeddify:
			result= np.zeros((1,sz),dtype=np.float32)
		else:
			result=[]

		i=0
		j=0
		for feat, sz in feat2sz.items():
			if embeddify:
				result[j:j+sz]= float(self.decode( categories[i], feat, apply_embeddings=embeddify))
			else:
				result += self.decode( categories[i], feat, false)

			i+=1
			j=sz 
		
		return result

	def feature(self, ds_type, feature, decode=False):
		'''
			Returns a feature from data
				this process may have losses since some on the lexicon are mapped to the same lemma
				(That applies to column LEMMA)

			args:
				ds_type 		.: string dataset name in (['train', 'valid', 'test'])
				feature	    .: string representing db column name
				decode      .: bool representing the better encoding

			returns:
				index  .: int<T> index column on the db
				values .: int<T> values 
		
		'''
		if not(ds_type in ['train', 'valid', 'test']):
			buff= 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
			raise ValueError(buff)
		else:		
			if ds_type in ['train']:
				lb=0
				ub=conf.DATASET_TRAIN_SIZE 
			elif ds_type in ['valid']:
				lb=conf.DATASET_TRAIN_SIZE 
				ub=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE
			else:
				lb=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE 
				ub=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE + conf.DATASET_TEST_SIZE
			if decode:
				d={idx: self.decode(self.data[feature][idx],feature)					
					for idx, p in self.data['P'].items()
						if p>=lb and p< ub+1}
			else:
				#unzips features
				d=  { idx: self.data[feature][idx]
					for idx, p in self.data['P'].items()
						if p>=lb and p< ub+1}
		return d



	def iterator(self, ds_type, decode=False):
		if not(ds_type in ['train', 'valid', 'test']):
			buff= 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
			raise ValueError(buff)
		else:
			as_dict=True
			fn = lambda x : self.sequence_obsexample(x, self.features, as_dict, decode)
			if ds_type in ['train']:
				lb=0
				ub=conf.DATASET_TRAIN_SIZE 
			elif ds_type in ['valid']:
				lb=conf.DATASET_TRAIN_SIZE 
				ub=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE
			else:
				lb=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE 
				ub=conf.DATASET_TRAIN_SIZE + conf.DATASET_VALID_SIZE + conf.DATASET_TEST_SIZE


		low=-1
		high=-1		
		prev_idx=-1
		for idx, prop in list(zip(*self.feature('P'))):
			if prop > lb and low==-1:
				low= idx 
			
			if prop > ub and high==-1:				
				high= prev_idx
				break
			prev_idx=idx

		if high==-1:
			high=prev_idx

		return _PropbankIterator(low, high, fn)

	def _feature_size(self, feat):	
		'''
			Returns the dimension of the feature with feature name feat
			args:
				features : str  feature name

			returns:
				sz 			.: int indicating the total feature size
		'''
		sz=0
		if conf.META[feat] in ['txt']:
			sz+=len(self.embeddings[0])
		elif conf.META[feat] in ['hot']:
			sz+=len(self.onehot[feat])
		else:
			sz+=1

		return sz
if __name__ == '__main__':
	PROP_DIR = '{:}/datasets/binaries/'.format(ROOT_DIR)
	PROP_PATH = '{:}/{:}'.format(PROP_DIR, 'db_pt_LEMMA_glove_s50.pickle')
	
	# propbank = Propbank()	
	# propbank.define()
	# propbank.persist(PROP_DIR)
	propbank = Propbank.recover(PROP_PATH)		
	# import code; code.interact(local=dict(globals(), **locals()))		

	# PRED_d = propbank.feature('valid', 'PRED', True) # text
	# M_R_d = propbank.feature('valid', 'M_R', True) # numerical
	ARG_d = propbank.feature('valid', 'ARG', True) # categorical
	# import code; code.interact(local=dict(globals(), **locals()))		