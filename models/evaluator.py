'''
	Created on Mar 19, 2018
	
	@author: Varela

'''

class Evaluator(object):
	'''
		Provides common machine learning avaluation functionality
			* accuracy
			* precision
			* recall
			* f1
			* confusion matrix
	'''
	def __init__(self, T):
		'''
			args:
	  		T       .:	dict<int,str> ground truth labels
		'''				
		self.T= T

	def accuracy(self, predictions):
		successes= [self.T[idx] == predictions[idx]
			for idx in predictions]
		return float(sum(successes)) / len(successes)
	
	def confusion_matrix(self, predictions):
		raise NotImplementedError('confusion_matrix not implemented')

	def precision(self, predictions):
		raise NotImplementedError('precision not implemented')

	def recall(self, predictions):
		raise NotImplementedError('recall not implemented')	

	def f1_score(self, predictions):
		raise NotImplementedError('f1_score not implemented')

