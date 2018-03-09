'''
Created on Mar 02, 2018
	@author: Varela
	
	implementation of (Zhou,et Xu, 2015)
	
	ref:
	http://www.aclweb.org/anthology/P15-1109

'''

class LanguageModel(object):
	'''
	The language model 


	1. for NER entities replace by the tag organization, person, location
	2. for smaller than 5 tokens replace by one hot encoding 
	3. include time i.e 20h30, 9h in number embeddings '0'
	4. include ordinals 2º 2ª in number embeddings '0'
	5. include tel._38-4048 in numeber embeddings '0'
	New Word embedding size = embedding size + one-hot enconding of 2	


	'''	
	def __init__(self):
	# load corpus
	# apply
	# load exception files on /corpus_exceptions
	raise NotImplementedError
	
	def encode(word, feature):
		'''
			Returns feature numeric representation
			args:

			returns:
				list containing 
		'''
		raise NotImplementedError
	def decode(word, feature):
		'''
			Returns feature numeric representation
			args:

			returns:
				list containing 
		'''		
		raise NotImplementedError

	def size(features):	
		'''
			Returns the dimension of the features 
			args:
				features : list<str>  with the features names
		'''
		raise NotImplementedError


