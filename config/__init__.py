'''
Created on Mar 14, 2018
    @author: Varela

    Defines project wide constants

'''
# import tensorflow as tf
#INPUTS AND DATASETS
import yaml

INPUT_DIR = 'datasets/binaries/'
SCHEMA_DIR = 'datasets/schemas/'

BASELINE_DIR = 'datasets/baseline/'
LANGUAGE_MODEL_DIR = 'datasets/txts/embeddings/'

DATA_ENCODING = 'TKN'


def get_config(embs_model, lang='pt'):
    cnf_path = 'config/{}/{}.yaml'.format(lang, embs_model)
    cnf_dict = {}
    with open(cnf_path, mode='r') as f:
        cnf_dict = yaml.load(f)
    return cnf_dict


def set_config(cnf_dict, embs_model, lang='pt'):
    cnf_path = 'config/{}/{}.yaml'.format(lang, embs_model)
    with open(cnf_path, mode='w') as f:
        yaml.dump(cnf_dict, f)

    return cnf_dict


DATASET_PROPOSITION_DICT = {
    'pt': {
        '1.0': {
            'train': 5296,
            'valid': 239,
            'test': 239,
        },
        '1.1': {
            'train': 5099,
            'valid': 569,
            'test': 263,
        },
    },
    'en': {
        'train': 90751,
        'valid': 3248
    }
}

FEATURE_LABELS = ['ID', 'FORM', 'MARKER',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']

OPTIONAL_LABELS = ['GPOS', 'FORM_CTX_P-3', 'FORM_CTX_P-2',
                    'SHALLOW_CHUNKS', 'FORM_CTX_P+2', 'FORM_CTX_P+3']

TARGET_LABELS = ['R', 'T', 'IOB', 'ARG']

META_LABELS = ['P', 'PRED', 'INDEX']
