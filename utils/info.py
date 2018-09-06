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

def get_index(columns_list, columns_dims_dict, column_name):
    '''
        Returns column index from descriptor
        args:
            columns_list            .: list<str> input features + target
            columns_dims_dict        .: dict<str, int> holding the columns
            column_name             .:  str name of the column to get the index from

        returns:
    '''
    features_set = set(config.CATEGORICAL_FEATURES).union(config.EMBEDDED_FEATURES)
    used_set = set(columns_list)
    descriptor_list = sorted(list(features_set - used_set))
    index = 0
    for descriptor in descriptor_list:
        if descriptor == column_name:
            break
        else:
            index += columns_dims_dict[descriptor]
    return index


def get_dims(labels_list, labels_dim_dict):
    return sum([labels_dim_dict[label] for label in labels_list])


def get_binary(ds_type, embeddings, version='1.0'):
    if ds_type not in ('train', 'test', 'valid', 'deep'):
        raise ValueError('Invalid dataset label {:}'.format(ds_type))

    prefix = '' if ds_type in ('deep') else 'db'
    ext = 'pickle' if ds_type in ('deep') else 'tfrecords'
    dbname = '{:}{:}_{:}.{:}'.format(prefix, ds_type, embeddings, ext)
    return '{:}{:}/{:}'.format(INPUT_DIR, version, dbname)

# def get_db_bounds(ds_type, version='1.0'):
#     '''Returns upper and lower bound proposition for dataset

#     Dataset breakdowns are done by partioning of the propositions

#     Arguments:
#         ds_type {str} -- Dataset type this must be `train`, `valid`, `test`

#     Retuns:
#         bounds {tuple({int}, {int})} -- Tuple with lower and upper proposition
#             for ds_type

#     Raises:
#         ValueError -- [description]
#     '''
#     ds_tuple = ('train', 'valid', 'test',)
#     version_tuple = ('1.0', '1.1',)

#     if not(ds_type in ds_tuple):
#         _msg = 'ds_type must be in {:} got \'{:}\''
#         _msg = _msg.format(ds_tuple, ds_type)
#         raise ValueError(_msg)

#     if not(version in version_tuple):
#         _msg = 'version must be in {:} got \'{:}\''
#         _msg = _msg.format(version_tuple, version)
#         raise ValueError(_msg)
#     else:
#         size_dict = config.DATASET_PROPOSITION_DICT[version]

#     lb = 1
#     ub = size_dict['train']

#     if ds_type  == 'train':
#         return lb, ub
#     else:
#         lb += ub
#         ub += size_dict['valid']
#         if ds_type  == 'valid':
#             return lb, ub
#         elif ds_type  == 'test':
#             lb += ub
#             ub += size_dict['test']
#         return lb, ub
