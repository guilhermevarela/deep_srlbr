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
import re
import string
from gensim.models import KeyedVectors

EMBEDDINGS_DIR= './datasets/embeddings/'
CORPUS_EXCEPTIONS_DIR= './datasets/corpus_exceptions/'
CSVS_DIR= './datasets/csvs/'

def _fetch_word2vec(natural_language_embedding_file):
	print('Fetching word2vec...')
	try:		
		embedding_path= '{:}{:}.txt'.format(EMBEDDINGS_DIR, natural_language_embedding_file)		
		word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")			
	except FileNotFoundError:
		print('natural_language_feature: {:} not found .. reverting to \'glove_s50\' instead'.format(natural_language_embedding_file))
		natural_language_embedding_file='glove_s50'
		embedding_path= '{:}{:}.txt'.format(EMBEDDINGS_DIR, natural_language_embedding_file)		
		word2vec  = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")			
	finally:
			print('Fetching word2vec... done')	

	return word2vec

def _fetch_corpus_exceptions(corpus_exception_file):
	'''
		Returns a list of supported feature names

		args:
			corpus_exception_file

			ner_tag

		returns:
			features : list<str>  with the features names
	'''
	print('Fetching {:}...'.format(corpus_exception_file))
	corpus_exceptions_path = '{:}{:}'.format(CORPUS_EXCEPTIONS_DIR, corpus_exception_file)
	df = pd.read_csv(corpus_exceptions_path, sep='\t')
	print('Fetching {:}...done'.format(corpus_exception_file))	

	return set(df['TOKEN'].values)



def _preprocess(lexicon, word2vec):	
	# define outputs
	total_words= len(lexicon)
	lexicon2token = dict(zip(lexicon, ['unk']*total_words))

	# fetch exceptions list
	pers= _fetch_corpus_exceptions('corpus-word-missing-pers.txt')
	locs= _fetch_corpus_exceptions('corpus-word-missing-locs.txt')
	orgs= _fetch_corpus_exceptions('corpus-word-missing-orgs.txt')
	

	#define regex
	re_punctuation= re.compile(r'[{:}]'.format(string.punctuation), re.UNICODE)	
	re_number= re.compile(r'^\d+$')
	re_tel   = re.compile(r'^tel\._')
	re_time  = re.compile(r'^\d{1,2}h\d{0,2}$')	
	re_ordinals= re.compile(r'º|ª')	

	for word in list(lexicon):
		# some hiffenized words belong to embeddings
		# ex: super-homem, fim-de-semana, pré-qualificar, caça-níqueis
		token = word.lower() 
		if token in word2vec:					
			lexicon2token[word]= token
		else:
			# if word in ['Rede_Globo', 'Hong_Kong', 'Banco_Central']:
			# 	import code; code.interact(local=dict(globals(), **locals()))			
			token = re_tel.sub('', token)
			token = re_ordinals.sub('', token)
			token = re_punctuation.sub('', token)

			token = re_time.sub('0', token)
			token = re_number.sub('0', token)

			if token in word2vec:
				lexicon2token[word]= token.lower()
			else:	
				if word in pers:
					lexicon2token[word] = 'pessoa'
				else:	
					if word in orgs:
						lexicon2token[word] = 'organização'	
					else:
						if word in locs:
							lexicon2token[word] = 'local'				


	total_tokens=	len([value for value in  lexicon2token.values() if not(value == 'unk')])
	
	print('Preprocess finished. Found {:} of {:} words, missing {:.2f}%'.format(total_words,
	 total_tokens, 100*float(total_words-total_tokens)/ total_words))				

	return lexicon2token


class Test(object):
	def __init__(self):
		self.foo = 'foo'

	def foo(self):
		print(self.foo)

class LanguageModelPt:
	'''
	The language model 



	1. for NER entities within exception file
		 replace by the tag organization, person, location
	2. for smaller than 5 tokens replace by one hot encoding 
	3. include time i.e 20h30, 9h in number embeddings '0'
	4. include ordinals 2º 2ª in number embeddings '0'
	5. include tel._38-4048 in numeber embeddings '0'
	New Word embedding size = embedding size + one-hot enconding of 2	


	'''	
	# def __init__(self, natural_language_lexicon, natural_language_feature= ''):
	def __init__(self):
		'''
			Returns new instance with a loaded language model
			args:
				natural_language_lexicon 			 .:  set containing all natural language tokens

				natural_language_embedding_file .: str representing the path to the n.l. file model

			returns:
				lmpt														.: object an instance of the LanguageModelPt class
		'''
	
	path=  '{:}zhou.csv'.format(CSVS_DIR)
	df= pd.read_csv(path)

	natural_language_feature= 'LEMMA'
	natural_language_lexicon= set(df[natural_language_feature].values)
	natural_language_embedding_file='glove_s50'	
	word2vec= _fetch_word2vec(natural_language_embedding_file)

	
	# Preprocess
	total_words= len(set(natural_language_lexicon))	
	print('Processing total lexicon is {:}'.format(total_words));
	lexicon = list(natural_language_lexicon)
	
	

	self.lexicon2token = _preprocess(lexicon, word2vec)
	self.token2idx= {'unk':0}
	
	tokens= set(self.lexicon2token.values())
	idx=1
	for token in list(tokens):
		if not(token in self.token2idx):
			self.token2idx[token]=idx
			idx+=1


	def encode(self, word, feature):
		'''
			Returns feature representation
			args:
				word   		.: string token before embeddings or encoding strategies

				feature 	.: string feature name

			returns:				
				result    .: list<> with the numeric representation of word under feature
		'''
		raise NotImplementedError

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
		raise NotImplementedError

	def features(self):		
		'''
			Returns a list of supported feature names

			returns:
				features : list<str>  with the features names
		'''
		raise NotImplementedError
	
	def add(self, feature_lexicon, feature,  encoding_strategy=''):
		raise NotImplementedError

	def example(self, values):
		raise NotImplementedError




	


if __name__ == '__main__':
	# lmpt = LanguageModelPt()
	Test()
	# import code; code.interact(local=dict(globals(), **locals()))		