'''
Created on Aug 1, 2018
    @author: Varela

    Defines project wide constants

'''
import argparse
import os, glob

import yaml
import pandas as pd

import config
import models.feature_factory as fac
from models.propbank_encoder import PropbankEncoder
from datasets import tfrecords_builder

SCHEMA_PATH = '{:}gs.yaml'.format(config.SCHEMA_DIR)

SHIFTS = (-3, -2, -1, 0, 1, 2, 3)

FEATURE_MAKER_DICT = {
    'chunks.csv': {'marker_fnc': lambda x, y: fac.process_chunk(x, version=y), 'column': 'chunk features'},
    'predicate_marker.csv': {'marker_fnc': lambda x, y: fac.process_predmarker(x, version=y), 'column': 'predicate marker feature'},
    'form.csv': {'marker_fnc': lambda x, y: fac.process_shifter_ctx_p(x, ['FORM'], SHIFTS, version=y), 'column': 'form predicate context features'},
    'gpos.csv': {'marker_fn': lambda x, y: fac.process_shifter_ctx_p(x, ['GPOS'], SHIFTS, version=y), 'column': 'gpos predicate context features'},
    'lemma.csv': {'marker_fnc': lambda x, y: fac.process_shifter_ctx_p(x, ['LEMMA'], SHIFTS, version=y), 'column': 'lemma predicate context features'},
    't.csv': {'marker_fnc': lambda x, y: fac.process_t(x, version=y), 'column': 'chunk label class'},
    'iob.csv': {'marker_fnc': lambda x, y: fac.process_iob(x, version=y), 'column': 'iob class'}
}


def make_propbank_encoder(encoder_name='deep_glo50', language_model='glove_s50', version='1.0', verbose=True):
    ''' Creates a ProbankEncoder instance from strings.

    Arguments:
        encoder_name:
        language_model:
        version
        verbose

    Returns:

    '''
    # Process inputs
    prefix_dir = config.LANGUAGE_MODEL_DIR
    prefix_target_dir = 'datasets/csvs/{:}/'.format(version)
    gs_path = '{:}gs.csv'.format(prefix_target_dir)
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

    dfgs = pd.read_csv(gs_path, index_col=0, sep=',', encoding='utf-8')

    column_files = [
        'column_chunks/chunks.csv',
        'column_predmarker/predicate_marker.csv',
        'column_shifts_ctx_p/form.csv',
        'column_shifts_ctx_p/gpos.csv',
        'column_shifts_ctx_p/lemma.csv',
        'column_t/t.csv',
        'column_iob/iob.csv'
    ]

    gs_dict = dfgs.to_dict()
    for column_filename in column_files:
        column_path = '{:}{:}'.format(prefix_target_dir, column_filename)
        if not os.path.isfile(column_path):

            *dirs, filename = column_path.split('/')
            # filename = column_path.split('/')[-1]
            # dirs = column_path.split('/')[:-1]
            dir_ = '/'.join(dirs)
            if not os.path.isdir(dir_):
                os.makedirs(dir_)

            column_dict = FEATURE_MAKER_DICT[filename]
            maker_fnc, msg = column_dict['marker_fnc'], column_dict['column']

            if verbose:
                print('processing:\t{:}'.format(msg))
            column_df = maker_fnc(gs_dict, version)
        else:
            column_df = pd.read_csv(column_path, index_col=0, encoding='utf-8')
        dfgs = pd.concat((dfgs, column_df), axis=1)

    propbank_encoder = PropbankEncoder(
        dfgs.to_dict(),
        schema_dict,
        language_model=language_model,
        dbname=encoder_name
    )
    if not os.path.isdir(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR)
    propbank_encoder.persist('datasets/binaries/', filename=encoder_name)
    return propbank_encoder


def make_tfrecords(encoder_name='deep_glo50', propbank_encoder=None):
    if propbank_encoder is None:
        propbank_encoder = PropbankEncoder.recover('datasets/binaries/{:}.pickle'.format(encoder_name))

    suffix = encoder_name.split('_')[-1]
    column_filters = None

    config_dict = propbank_encoder.columns_config  # SEE schemas/gs.yaml    
    for ds_type in ('test', 'valid', 'train'):
         iterator_ = propbank_encoder.iterator(ds_type, filter_columns=column_filters)
         tfrecords_builder(iterator_, ds_type, config_dict, suffix=suffix)


def get_model(mname):
    if mname == 'wang2vec':
        return 'wan'

    if mname == 'word2vec':
        return 'wrd'

    if mname == 'glove':
        return 'glo'

    return mname[:3]


def get_filename(file_path):
    '''Extracts filename from full path

    Arguments:
        file_path {str} -- sys path
    '''
    return f.split('/')[-1].replace('.txt', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Preprocess Deep SRL system features and embeddings''')

    parser.add_argument('language_model',
                        type=str, nargs='+', default='glove_s50',
                        help='''language model for embeddings, more info:
                             http://nilc.icmc.usp.br/embeddings''')

    parser.add_argument('-version', type=str,
                        dest='version', nargs=1, choices=('1.0', '1.1',),
                        help='PropBankBr: version 1.0 or 1.1')

    args = parser.parse_args()
    language_model = args.language_model[0]
    version = args.version[0]
    print(language_model, version)
    lmpath = '{:}{:}.txt'.format(config.LANGUAGE_MODEL_DIR, language_model)


    if not os.path.isfile(lmpath):
        if not os.path.isdir(config.LANGUAGE_MODEL_DIR):
            os.makedirs(config.LANGUAGE_MODEL_DIR)
            msg_ = '''{:}:created.
                    Download cd embeddings
                    http://nilc.icmc.usp.br/embeddings.'''.\
                format(config.LANGUAGE_MODEL_DIR)
            raise ValueError(msg_)
        else:
            glob_regex = '{:}*.txt'.format(config.LANGUAGE_MODEL_DIR)
            language_model_list = [
                get_filename(f) for f in glob.glob(glob_regex)]

            if language_model_list:
                raise ValueError('''{:}: not found.
                                 Some avalable options are {:}'''.
                                 format(language_model, language_model_list))
    else:
        model_, sz_ = language_model.split('_s')


    encoder_name = 'deep_{:}{:}'.format(get_model(model_), sz_)
    propbank_encoder = make_propbank_encoder(
        encoder_name=encoder_name,
        language_model='glove_s50'
    )
    make_tfrecords(encoder_name='deep_glo50') 