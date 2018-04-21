'''
Created on Mar 14, 2018
	@author: Varela

	Defines project wide constants

'''

#INPUTS AND DATASETS
INPUT_DIR = 'datasets/binaries/'
SCHEMA_DIR = 'datasets/schemas/'
LANGUAGE_MODEL_DIR = 'datasets/txts/embeddings/'
DATASET_TRAIN_PATH= '{:}dbtrain_pt.tfrecords'.format(INPUT_DIR)
DATASET_VALID_PATH= '{:}dbvalid_pt.tfrecords'.format(INPUT_DIR)
DATASET_TEST_PATH = '{:}dbtest_pt.tfrecords'.format(INPUT_DIR)

DATASET_SIZE= 5931
DATASET_TRAIN_SIZE= 5099
DATASET_VALID_SIZE= 569
DATASET_TEST_SIZE=  263

#FEATURES
SEQUENCE_FEATURES=      [ 'INDEX', 'ID', 'S', 'P', 'P_S', 
	'LEMMA', 'GPOS', 'MORF',  'DTREE',     'FUNC', 
	'CTREE', 'PRED',  'ARG', 'CTX_P-3', 'CTX_P-2', 
	'CTX_P-1', 'CTX_P+1', 'CTX_P+2', 'CTX_P+3',  'M_R', 
	'PRED_1', 'T']

SEQUENCE_FEATURES_TYPES=['int', 'int', 'int', 'int', 'int',  
	'txt', 'hot', 'hot', 'hot', 'hot', 
	'hot', 'txt', 'hot',  'txt', 'txt', 
	'txt', 'txt',  'txt', 'txt',  'int',  
	'txt', 'hot']

DEFAULT_INPUT_SEQUENCE_FEATURES= ['ID', 'LEMMA', 'M_R', 'PRED_1', 'CTX_P-1', 'CTX_P+1']
DEFAULT_OUTPUT_SEQUENCE_TARGET= 'T'

META= dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))
