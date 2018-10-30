'''
Created on Aug 1, 2018
    @author: Varela

    Defines project wide constants

'''
import argparse
import os
import glob
import re
import yaml
import pandas as pd

import config
import models.feature_factory as fac
from models.propbank_encoder import PropbankEncoder
from datasets import tfrecords_builder
from utils.info import get_binary

SCHEMA_PATH = '{:}gs.yaml'.format(config.SCHEMA_DIR)

SHIFTS = (-3, -2, -1, 0, 1, 2, 3)

FEATURE_MAKER_DICT = {
    'argrecon.csv': {'marker_fnc': lambda x, l, y: fac.process_argrecon(x, lang=l, version=y), 'column': 'argument recognition'},
    'chunks.csv': {'marker_fnc': lambda x, l, y: fac.process_chunk(x, lang=l, version=y), 'column': 'chunk features'},
    'ctree_chunk.csv': {'marker_fnc': lambda x, l, y: fac.process_ctreechunk(x, lang=l, version=y), 'column': 'shallow chunk features'},
    'predicate_marker.csv': {'marker_fnc': lambda x, l, y: fac.process_predmarker(x, lang=l, version=y), 'column': 'predicate marker feature'},
    'form.csv': {'marker_fnc': lambda x, l, y: fac.process_shifter_ctx_p(x, ['FORM'], SHIFTS, lang=l, version=y), 'column': 'form predicate context features'},
    'gpos.csv': {'marker_fnc': lambda x, l, y: fac.process_shifter_ctx_p(x, ['GPOS'], SHIFTS, lang=l, version=y), 'column': 'gpos predicate context features'},
    'lemma.csv': {'marker_fnc': lambda x, l, y: fac.process_shifter_ctx_p(x, ['LEMMA'], SHIFTS, lang=l, version=y), 'column': 'lemma predicate context features'},
    't.csv': {'marker_fnc': lambda x, l, y: fac.process_t(x, lang=l, version=y), 'column': 'chunk label class'},
    'iob.csv': {'marker_fnc': lambda x, l, y: fac.process_iob(x, lang=l, version=y), 'column': 'iob class'}
}


def make_propbank_encoder(encoder_name='deep_glo50',
                          language_model='glove_s50',
                          lang='pt',
                          version='1.0',
                          verbose=True):
    '''Creates a ProbankEncoder instance from strings.

    [description]

    Keyword Arguments:
        encoder_name {str} -- [description] (default: {'deep_glo50'})
        language_model {str} -- [description] (default: {'glove_s50'})
        lang {str} -- [description] (default: {'pt'})
        version {str} -- [description] (default: {'1.0'})
        verbose {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    '''

    # Process inputs
    prefix_dir = '{:}{:}/'.format(config.LANGUAGE_MODEL_DIR, lang)
    embs_model, embs_size = language_model.split('_s')
    if lang == 'pt':
        prefix_target_dir = 'datasets/csvs/pt/{:}/'.format(version)
        file_path = '{:}{:}.txt'.format(prefix_dir, language_model)
    else:
        prefix_target_dir = 'datasets/csvs/en/'
        file_path = '{:}{:}.6B.{:}d.txt'.format(prefix_dir, embs_model, embs_size)

    gs_path = '{:}gs.csv'.format(prefix_target_dir)


    if not os.path.isfile(file_path):
        glob_regex = '{:}*'.format(prefix_dir)
        options_list = [
            re.sub('\.txt','', re.sub(prefix_dir,'', file_path))
            for file_path in glob.glob(glob_regex)]
        _errmsg = '{:} not found avalable options are in {:}'
        raise ValueError(_errmsg.format(language_model, options_list))




    # Getting to the schema    
    with open(SCHEMA_PATH, mode='r') as f:
        schema_dict = yaml.load(f)

    dfgs = pd.read_csv(gs_path, index_col=0, sep=',', encoding='utf-8')

    column_files = [
        'column_argrecon/argrecon.csv',
        'column_chunks/chunks.csv',
        'column_ctree_chunks/ctree_chunk.csv',
        'column_predmarker/predicate_marker.csv',
        'column_shifts_ctx_p/form.csv',
        'column_shifts_ctx_p/gpos.csv',
        'column_t/t.csv',
        'column_iob/iob.csv'
    ]
    if lang == 'pt':
        column_files.append('column_shifts_ctx_p/lemma.csv')
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
            maker_fnc(gs_dict, lang, version)


        column_df = pd.read_csv(column_path, index_col=0, encoding='utf-8')
        dfgs = pd.concat((dfgs, column_df), axis=1)

    propbank_encoder = PropbankEncoder(
        dfgs.to_dict(),
        schema_dict,
        language_model=language_model,
        lang=lang,
        dbname=encoder_name,
        version=version
    )

    model_alias = '{:}{:}'.format(get_model_alias(embs_model), embs_size)

    bin_path = get_binary('deep', model_alias, version=version)
    bin_dir = '/'.join(bin_path.split('/')[:-1]) + '/'
    if not os.path.isdir(bin_dir):
        os.makedirs(bin_dir)

    propbank_encoder.persist(bin_dir, filename=encoder_name)
    return propbank_encoder


def make_tfrecords(encoder_name='deep_glo50',
                   propbank_encoder=None,
                   version='1.0',
                   lang='pt'):

    # PREPARE WRITE DIRECTORIES
    embs_model = encoder_name.split('_')[-1]
    bin_dir = 'datasets/binaries/{:}/'.format(lang)
    if lang == 'pt':
        bin_dir += '{:}/'.format(version)

    if version in ('1.0') or lang == 'en':
        bin_dir += '{:}/'.format(embs_model)

    if not os.path.isdir(bin_dir):
        os.makedirs(bin_dir)

    if propbank_encoder is None:
        bin_path = '{:}{:}.pickle'.format(bin_dir, encoder_name)
        propbank_encoder = PropbankEncoder.recover(bin_path)

    cnf_dict = propbank_encoder.to_config(config.DATA_ENCODING)
    config.set_config(cnf_dict, embs_model, lang=lang)

    flt = None
    encoding = config.DATA_ENCODING

    if lang == 'pt':
        ds_tuple = ('test', 'valid', 'train')
    else:
        ds_tuple = ('valid', 'train')

    for ds_type in ds_tuple:
        iter_ = propbank_encoder.iterator(ds_type, filter_columns=flt, encoding=encoding)
        tfrecords_builder(iter_, ds_type, embs_model, lang=lang, version=version, encoding=encoding)


def get_model_alias(mname):
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
                        type=str, default='glove_s50',
                        help='''language model for embeddings, more info:
                             http://nilc.icmc.usp.br/embeddings''')

    # This argument now is an environment variable in config module
    # parser.add_argument('--encoding', type=str, dest='encoding',
    #                     choices=('HOT', 'EMB', 'TKN', 'IDX'), default='EMB',
    #                     help='''Choice of feature representation based on column type --
    #                     `int`, `bool`, `text`, `choice`. `TKN` will keep `text`
    #                     features as index to be embedded for the input pipeline
    #                     and will one-hot `choice` values. `EMB` will embed `text`
    #                     features and will one-hot encode `choice` features.''')

    parser.add_argument('--lang', type=str, dest='lang',
            			choices=('en', 'pt'), default='pt',
			            help='PropBank language')


    parser.add_argument('--version', type=str, dest='version',
                        choices=('1.0', '1.1',), default='1.0',
                        help='''PropBankBr: version 1.0 or 1.1
                                only active if lang=`pt`''')

    args = parser.parse_args()
    language_model = args.language_model
    version = args.version
    lang = args.lang

    print(language_model, lang, version)
    lmpath = '{:}{:}/{:}.txt'.format(config.LANGUAGE_MODEL_DIR, lang, language_model)


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

    embs_model, embs_size = language_model.split('_s')

    encoder_name = 'deep_{:}{:}'.format(
        get_model_alias(embs_model), embs_size)

    propbank_encoder = make_propbank_encoder(
        encoder_name=encoder_name,
        language_model=language_model,
        lang=lang,
        version=version
    )

    make_tfrecords(
        encoder_name=encoder_name,
        propbank_encoder=propbank_encoder,
        version=version,
        lang=lang
    )
