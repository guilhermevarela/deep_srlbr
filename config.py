'''
Created on Mar 14, 2018
	@author: Varela

	Defines project wide constants

'''

SEQUENCE_FEATURES=      [
	'ID', 'S', 'P', 'P_S', 'LEMMA', 'GPOS', 'MORF', 'DTREE', 'FUNC',
	'CTREE', 		 'PRED', 			'ARG', 'CTX_P+1', 'CTX_P-3', 'CTX_P-2', 
	'CTX_P-1', 'CTX_P+2', 'CTX_P+3',  		'M_R', 'PRED_1', 'T']

SEQUENCE_FEATURES_TYPES=['int', 'int', 'int', 'int',   'txt', 'hot', 'hot', 'hot', 'hot',
	'hot', 	'txt', 'hot',    'txt',    'txt',    'txt',    'txt',    
	'txt', 'txt',  'int',  'txt', 'hot']

DEFAULT_INPUT_SEQUENCE_FEATURES= ['LEMMA', 'M_R', 'PRED_1', 'CTX_P-1', 'CTX_P+1']
DEFAULT_OUTPUT_SEQUENCE_TARGET= 'T'

META= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))

DATASET_SIZE= 5931
DATASET_TRAIN_SIZE= 5099
DATASET_VALID_SIZE= 569
DATASET_TEST_SIZE=  263

TF_SEQUENCE_FEATURES= {key:tf.VarLenFeature(tf.int64) 
	for key in SEQUENCE_FEATURES
}

TF_CONTEXT_FEATURES=	{
	'T': tf.FixedLenFeature([], tf.int64)			
}
