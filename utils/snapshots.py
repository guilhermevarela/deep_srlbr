import datetime
import os
import json

from config import INPUT_DIR

def snapshot_hparam_string(embeddings_model='glo50', target_labels='T',
                      is_batch=True, learning_rate=5 * 1e-3,
                      version='1.0',hidden_layers=[16] * 4, **kwargs):
    '''Makes a nested directory to record model's data

    Stores the parameters of a given experiment within a nested
    directory structure -- the inner directory will be a timestamp

    Keyword Arguments:
        embeddings_model {str} -- Word embeddings mneumonic (default: {'glo50'})
                `glo`, `wan` , `wrd` for GloVe, Wang2Vec and Word2Vec.
        target_labels {str} -- Target label (default: {'T'})
        is_batch {bool} -- If true performs batch training (default: {True})
        learning_rate {float} -- Model's  learning rate (default: {5 * 1e-3})
        hidden_layers {list{int}} -- list of integers (default: {[16] * 4})
        ctx_p {int} -- Moving window around predicate (default: {1})

    Returns:
        snapshot_dir {str} -- [description]
    '''
    param_list = [None] * 7
    params_dict = dict(locals())
    params_dict.update(kwargs)
    for key, value in params_dict.items():
        if value is not None:
            if key == 'version':
                param_list[0] = value

            if key == 'embeddings_model':
                param_list[1] = value

            if key == 'hidden_layers':
                hidden_list_ = [str(s) for s in value]
                param_list[2] = 'x'.join(hidden_list_)
                param_list[2] = 'hs_{:}'.format(param_list[2])

            if key == 'ctx_p':
                param_list[3] = 'ctxp_{:d}'.format(value)

            if key == 'target_labels':
                param_list[4] = '_'.join(target_labels)

            if key == 'is_batch':
                param_list[5] = 'batch' if value else 'kfold'

            if key == 'learning_rate':
                param_list[6] = 'lr_{:.2e}'.format(value)

    snapshot_dir = ''
    for param_ in param_list:
        snapshot_dir += '/{}'.format(param_)

    snapshot_dir += '/'

    return snapshot_dir

def snapshot_persist(target_dir,  **kwargs):
# def snapshot_persist(target_dir, input_labels=None, target_labels=None,
#                     hidden_layers=None, embeddings=None,
#                     epochs=None, lr=None, batch_size=None,
#                     kfold=None, version='1.0', **kwargs):
    '''Saves the parameters for a given model 
    
    Creates a directory it one doesnot exist
    Saves the contents for the parameters in a timestamp dir

    Arguments:
        target_dir {[type]} -- [description]
        **kwargs {[type]} -- [description]

    Keyword Arguments:
        input_labels {[type]} -- [description] (default: {None})
        target_labels {[type]} -- [description] (default: {None})
        hidden_layers {[type]} -- [description] (default: {None})
        embeddings {[type]} -- [description] (default: {None})
        epochs {[type]} -- [description] (default: {None})
        lr {[type]} -- [description] (default: {None})
        batch_size {[type]} -- [description] (default: {None})
        version {str} -- [description] (default: {'1.0'})

    Returns:
        target_dir [type] -- Updated target diretory
    '''
    timestamp = datetime.datetime.utcnow()
    timestamp = timestamp.strftime('%Y-%m-%d %H%M%S')
    target_dir += '{:}/'.format(timestamp)

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    target_path = '{:}params.json'.format(target_dir)

    KEYS = {'input_labels', 'target_labels',
            'hidden_layers', 'embeddings_model',
            'embeddings_trainable', 'epochs',
            'lr', 'batch_size', 'kfold', 'version',
            'ru', 'chunks', 'r_depth'}

    # Clear exclusve parameters
    if 'kfold' in kwargs:
        keys_set = KEYS - {'batch_size'}
    else:
        keys_set = KEYS - {'kfold'}

    if keys_set < kwargs.keys():  # issubset
        keys_list = sorted(list(keys_set))
    else:
        raise KeyError('Missing items: {:}'.format(keys_set - kwargs.keys()))

    params_dict = {key: kwargs[key] for key in keys_list}

    with open(target_path, mode='w') as f:
        json.dump(params_dict, f)

    return target_dir


def snapshot_recover(target_dir):
    target_path = '{:}params.json'.format(target_dir)
    with open(target_path, mode='r') as f:
        params_dict = json.load(f)

    return params_dict