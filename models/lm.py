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
	#prepare output
	lexicon2token = dict(zip(natural_language_lexicon, ['unk']*total_words))
	
	for word in lexicon:		
		if word.lower() in word2vec:
			lexicon2token[word]= word.lower()
		else:
			not_found.append(word)

	print('Found {:} of {:} --> missing (\%):{:2f}\%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	print('Processing removing punctuations')
	re_punctuation= re.compile(r'[{:}]'.format(string.punctuation), re.UNICODE)	
	aux=[] # doesnt deal with delete while iterates
	for i, word in enumerate(not_found):		
		token=  re_punctuation.sub('', word.lower())
		if token in word2vec:
			lexicon2token[word]= token
		else:
			aux.append(word)	

	not_found= aux
	aux=[]
	print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	# Maps numbers plain numbers, telefone numbers to zero		
	re_number= re.compile(r'^\d+$')
	re_tel   = re.compile(r'^tel\._')
	re_time  = re.compile(r'^\d{1,2}h\d{0,2}$')	
	re_ordinals= re.compile(r'º|ª')

	print('Processing dealing with numbers')
	
	
	for i, word in enumerate(not_found):		
		token=  re_tel.sub('', word.lower())			
		token=  re_punctuation.sub('', token)
		token=  re_ordinals.sub('', token)			
		token=  re_time.sub('0', token)			
		token= re_number.sub('0', token)

		if token in word2vec:
			lexicon2token[word]= token
		else:
			aux.append(word)
	not_found= aux
	aux=[]
	print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	
	print('Handling locales')
	locs= _fetch_corpus_exceptions('corpus-word-missing-locs.txt')
	for word in list(locs):
		lexicon2token[word]= 'local'
		not_found.remove(word)
	print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	
	
	print('Handling organizations')
	orgs= _fetch_corpus_exceptions('corpus-word-missing-orgs.txt')
	for word in list(orgs):
		lexicon2token[word]= 'organização'
		not_found.remove(word)

	# not_found= [word for word in not_found if not(word in aux)]	
	# aux=[]
	print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));		
	
	print('Handling people')
	pers= _fetch_corpus_exceptions('corpus-word-missing-pers.txt')
	for word in list(pers):
		lexicon2token[word]= 'pessoa'
		not_found.remove(word)
	
	# not_found= [word for word in not_found if not(word in aux)]	
	# aux=[]
	print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	return lexicon2token


class LanguageModelPt(object):
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
	# lexicon2token = dict(zip(natural_language_lexicon, ['unk']*total_words))
	print('Processing total lexicon is {:}'.format(total_words));
	lexicon = list(natural_language_lexicon)
	not_found=[]
	lexicon2token = _preprocess(lexicon, word2vec)
	# for word in lexicon:		
	# 	if word.lower() in word2vec:
	# 		lexicon2token[word]= word.lower()
	# 	else:
	# 		not_found.append(word)

	# print('Found {:} of {:} --> missing (\%):{:2f}\%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	# print('Processing removing punctuations')
	# re_punctuation= re.compile(r'[{:}]'.format(string.punctuation), re.UNICODE)	
	# aux=[] # doesnt deal with delete while iterates
	# for i, word in enumerate(not_found):		
	# 	token=  re_punctuation.sub('', word.lower())
	# 	if token in word2vec:
	# 		lexicon2token[word]= token
	# 	else:
	# 		aux.append(word)	

	# not_found= aux
	# aux=[]
	# print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	

	# # Maps numbers plain numbers, telefone numbers to zero		
	# re_number= re.compile(r'^\d+$')
	# re_tel   = re.compile(r'^tel\._')
	# re_time  = re.compile(r'^\d{1,2}h\d{0,2}$')	
	# re_ordinals= re.compile(r'º|ª')

	# print('Processing dealing with numbers')
	
	
	# for i, word in enumerate(not_found):		
	# 	token=  re_tel.sub('', word.lower())			
	# 	token=  re_punctuation.sub('', token)
	# 	token=  re_ordinals.sub('', token)			
	# 	token=  re_time.sub('0', token)			
	# 	token= re_number.sub('0', token)

	# 	if token in word2vec:
	# 		lexicon2token[word]= token
	# 	else:
	# 		aux.append(word)
	# not_found= aux
	# aux=[]
	# print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	



	
	# print('Handling locales')
	# locs= _fetch_corpus_exceptions('corpus-word-missing-locs.txt')
	# for word in list(locs):
	# 	lexicon2token[word]= 'local'
	# 	not_found.remove(word)
	# print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	
	
	# print('Handling locales')
	# orgs= _fetch_corpus_exceptions('corpus-word-missing-orgs.txt')
	# for word in list(orgs):
	# 	lexicon2token[word]= 'organização'
	# 	not_found.remove(word)

	# # not_found= [word for word in not_found if not(word in aux)]	
	# # aux=[]
	# print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));		
	
	# pers= _fetch_corpus_exceptions('corpus-word-missing-pers.txt')
	# for word in list(pers):
	# 	lexicon2token[word]= 'pessoa'
	# 	not_found.remove(word)
	
	# # not_found= [word for word in not_found if not(word in aux)]	
	# # aux=[]
	# print('Found {:} of {:} --> missing (%):{:0.2f}%'.format(total_words - len(not_found), total_words, 100*float(len(not_found))/ total_words));	
	import code; code.interact(local=dict(globals(), **locals()))			

			

	

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
	# path=  '{:}zhou.csv'.format(CSVS_DIR)
	# df= pd.read_csv(path)

	# lexicon= set(df['LEMMA'].values)
	lmpt = LanguageModelPt();