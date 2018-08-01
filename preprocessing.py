'''
Created on Aug 1, 2018
    @author: Varela

    Defines project wide constants

'''

import os

import yaml
import pandas as pd

import config
import models.feature_factory as fac
from models.propbank_encoder import PropbankEncoder

SCHEMA_PATH = '{:}gs.yaml'.format(config.SCHEMA_DIR)

SHIFTS = (-3, -2, -1, 0, 1, 2, 3)

FEATURE_MAKER_DICT = {
    'chunks.csv': lambda x: fac.process_chunk(x),
    'predicate_marker.csv': lambda x: fac.process_predmarker(x),
    'form.csv': lambda x: fac.process_shifter_ctx_p(x, ['FORM'], SHIFTS),
    'gpos.csv': lambda x: fac.process_shifter_ctx_p(x, ['GPOS'], SHIFTS),
    'lemma.csv': lambda x: fac.process_shifter_ctx_p(x, ['LEMMA'], SHIFTS),
    't.csv': lambda x: fac.process_t(x),
    'iob.csv': lambda x: fac.process_iob(x)
}


def make_propbank_encoder(encoder_name='deep_glo50', language_model='glove_s50'):
    ''' Creates a ProbankEncoder instance from strings.

    :param encoder_name:
    :param language_model:
    :return:
    '''
    # Process inputs
    prefix_dir = config.LANGUAGE_MODEL_DIR
    file_path = '{:}{:}.txt'.format(prefix_dir, language_model)

    if not os.path.isfile(file_path):
        glob_regex = '{:}*'.format(prefix_dir)
        options_list = [
            re.sub('\.txt','', re.sub(prefix_dir,'', file_path))
            for file_path in glob.glob(glob_regex)]
        _errmsg = '{:} not found avalable options are in {:}'
        raise ValueError(_errmsg.format(language_model ,options_list))




    # Getting to the schema
    with open(SCHEMA_PATH, mode='r') as f:        
        schema_dict = yaml.load(f)

    dfgs = pd.read_csv('datasets/csvs/gs.csv', index_col=0, sep=',', encoding='utf-8')

    column_files = [
        'datasets/csvs/column_chunks/chunks.csv',
        'datasets/csvs/column_predmarker/predicate_marker.csv',
        'datasets/csvs/column_shifts_ctx_p/form.csv',
        'datasets/csvs/column_shifts_ctx_p/gpos.csv',
        'datasets/csvs/column_shifts_ctx_p/lemma.csv',
        'datasets/csvs/column_t/t.csv',
        'datasets/csvs/column_iob/iob.csv'
    ]

    gs_dict = dfgs.to_dict()
    for column_path in column_files:
        if not os.path.isfile(column_path):
            *dirs, filename = column_path.split('/')
            dir_ = '/'.join(dirs)
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
            
            maker_fnc = FEATURE_MAKER_DICT[filename]
            column_df = maker_fnc(gs_dict)
        else:
            column_df = pd.read_csv(column_path, index_col=0, encoding='utf-8')
    
        dfgs = pd.concat((dfgs, column_df), axis=1)
    
    import code; code.interact(local=dict(globals(), **locals()))
    propbank_encoder = PropbankEncoder(dfgs.to_dict(),  schema_dict, language_model=language_model, dbname=encoder_name)
    # import code; code.interact(local=dict(globals(), **locals()))
    propbank_encoder.persist('datasets/binaries/', filename=encoder_name)
    return propbank_encoder



if __name__ == '__main__':
    # encoding_name = 'deep_wan50'
    # language_model = 'wang2vec_s50'
    # 
    # encoder_name = 'deep_wan50'    
    # language_model ='wang2vec_s50'

    # encoder_name = 'deep_wrd50'
    # language_model ='word2vec_s50'
    propbank_encoder = make_propbank_encoder(encoder_name='deep_glo50', language_model='glove_s50')