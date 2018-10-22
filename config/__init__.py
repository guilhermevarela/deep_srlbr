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

DATA_ENCODING = 'EMB'


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

