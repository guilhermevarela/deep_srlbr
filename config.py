'''
Created on Mar 14, 2018
    @author: Varela

    Defines project wide constants

'''
import tensorflow as tf
#INPUTS AND DATASETS
INPUT_DIR = 'datasets/binaries/'
SCHEMA_DIR = 'datasets/schemas/'

BASELINE_DIR = 'datasets/baseline/'
LANGUAGE_MODEL_DIR = 'datasets/txts/embeddings/'
DATASET_TRAIN_PATH = '{:}dbtrain_pt.tfrecords'.format(INPUT_DIR)
DATASET_VALID_PATH = '{:}dbvalid_pt.tfrecords'.format(INPUT_DIR)
DATASET_TEST_PATH = '{:}dbtest_pt.tfrecords'.format(INPUT_DIR)

DATASET_TRAIN_V2_PATH = '{:}dbtrain_pt_v2.tfrecords'.format(INPUT_DIR)
DATASET_VALID_V2_PATH = '{:}dbvalid_pt_v2.tfrecords'.format(INPUT_DIR)
DATASET_TEST_V2_PATH = '{:}dbtest_pt_v2.tfrecords'.format(INPUT_DIR)

DATASET_TRAIN_GLO50_PATH = '{:}dbtrain_glo50.tfrecords'.format(INPUT_DIR)
DATASET_VALID_GLO50_PATH = '{:}dbvalid_glo50.tfrecords'.format(INPUT_DIR)
DATASET_TEST_GLO50_PATH = '{:}dbtest_glo50.tfrecords'.format(INPUT_DIR)

DATASET_SIZE = 5931
DATASET_TRAIN_SIZE = 5099
DATASET_VALID_SIZE = 569
DATASET_TEST_SIZE = 263


DATASET_PROPOSITION_DICT = {
    '1.0': {
        'train': 5296,
        'valid': 239,
        'test': 239,
    },
    '1.1': {
        'train': 5099,
        'valid': 569,
        'test': 263,
    }
}
SEQUENCE_FEATURES = [ 'INDEX', 'ID', 'S', 'P', 'P_S',
    'LEMMA', 'GPOS', 'MORF',  'DTREE',     'FUNC', 
    'CTREE', 'PRED',  'ARG', 'CTX_P-3', 'CTX_P-2', 
    'CTX_P-1', 'CTX_P+1', 'CTX_P+2', 'CTX_P+3',  'MARKER', 
    'PRED_1', 'T']


CATEGORICAL_FEATURES = sorted([
    'ARG', 'GPOS', 'HEAD', 'ID', 'INDEX',
    'GPOS_CTX_P+0', 'GPOS_CTX_P+1', 'GPOS_CTX_P-1',
    'IOB', 'P', 'MARKER', 'T', 'SHALLOW_CHUNKS'
])

EMBEDDED_FEATURES = sorted([
    'FORM', 'FORM_CTX_P+0', 'FORM_CTX_P+1', 'FORM_CTX_P-1',
    'FORM_CTX_P+2', 'FORM_CTX_P+3', 'FORM_CTX_P-2', 'FORM_CTX_P-3',
    'LEMMA', 'LEMMA_CTX_P+0', 'LEMMA_CTX_P+1', 'LEMMA_CTX_P-1'
])


SEQUENCE_FEATURES_TYPES=['int', 'int', 'int', 'int', 'int',  
    'txt', 'hot', 'hot', 'hot', 'hot', 
    'hot', 'txt', 'hot',  'txt', 'txt', 
    'txt', 'txt',  'txt', 'txt',  'int',  
    'txt', 'hot']

DEFAULT_INPUT_SEQUENCE_FEATURES = ['ID', 'LEMMA', 'MARKER', 'PRED_1', 'CTX_P-1', 'CTX_P+1']
DEFAULT_OUTPUT_SEQUENCE_TARGET = 'T'

META = dict(zip(SEQUENCE_FEATURES, SEQUENCE_FEATURES_TYPES))
