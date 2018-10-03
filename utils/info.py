'''
    Created on Sep 06, 2018

    @author: Varela

    Provides information about tje dabases
'''

import config
from config import INPUT_DIR


def get_db_bounds(ds_type, version='1.0'):
    '''Returns upper and lower bound proposition for dataset

    Dataset breakdowns are done by partioning of the propositions

    Arguments:
        ds_type {str} -- Dataset type this must be `train`, `valid`, `test`

    Retuns:
        bounds {tuple({int}, {int})} -- Tuple with lower and upper proposition
            for ds_type

    Raises:
        ValueError -- [description]
    '''
    ds_tuple = ('train', 'valid', 'test',)
    version_tuple = ('1.0', '1.1',)

    if not(ds_type in ds_tuple):
        _msg = 'ds_type must be in {:} got \'{:}\''
        _msg = _msg.format(ds_tuple, ds_type)
        raise ValueError(_msg)

    if not(version in version_tuple):
        _msg = 'version must be in {:} got \'{:}\''
        _msg = _msg.format(version_tuple, version)
        raise ValueError(_msg)
    else:
        size_dict = config.DATASET_PROPOSITION_DICT[version]

    lb = 1
    ub = size_dict['train']
    if ds_type == 'train':
        return (lb, ub)
    else:
        lb = ub
        ub += size_dict['valid']
        if ds_type == 'valid':
            return (lb, ub)
        else:
            lb = ub
            ub += size_dict['test']
            return (lb, ub)


def get_binary(ds_type, embeddings_model, version='1.0'):
    if ds_type not in ('train', 'test', 'valid', 'deep'):
        raise ValueError('Invalid dataset label {:}'.format(ds_type))

    prefix = '' if ds_type in ('deep') else 'db'
    ext = 'pickle' if ds_type in ('deep') else 'tfrecords'
    dbname = '{:}{:}_{:}.{:}'.format(prefix, ds_type, embeddings_model, ext)

    prefix = '{:}{:}/'.format(INPUT_DIR, version)
    if version in ('1.0',):
        return '{:}{:}/{:}'.format(prefix, embeddings_model, dbname)
    else:
        return '{:}/{:}'.format(prefix, dbname)
