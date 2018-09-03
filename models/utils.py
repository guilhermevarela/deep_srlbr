'''
Created on Mar 12, 2018
    @author: Varela

    Fetching and preprocessing routines for models
    
    ref:
        https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py

    preprocessing rules:    
        Script used for cleaning corpus in order to train word embeddings.
        All emails are mapped to a EMAIL token.
        All numbers are mapped to 0 token.
        All urls are mapped to URL token.
        Different quotes are standardized.
        Different hiphen are standardized.
        HTML strings are removed.
        All text between brackets are removed.
        All sentences shorter than 5 tokens were removed.

'''
import re
import string
import datetime
import os

import pandas as pd
import config

from gensim.models import KeyedVectors
INPUT_DIR = 'datasets/binaries/'
EMBEDDINGS_DIR = 'datasets/txts/embeddings/'
CORPUS_EXCEPTIONS_DIR = 'datasets/txts/corpus_exceptions/'

def fetch_corpus(db_name):
    path=  '{:}{:}.csv'.format(CSVS_DIR, db_name)
    return pd.read_csv(path)

def fetch_word2vec(natural_language_embedding_file, verbose=True):
    if verbose:
        print('Fetching word2vec...')
    try:        
        embedding_path= '{:}{:}.txt'.format(EMBEDDINGS_DIR, natural_language_embedding_file)        
        word2vec = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")           
    except FileNotFoundError:
        if verbose:
            print('natural_language_feature: {:} not found .. reverting to \'glove_s50\' instead'.format(natural_language_embedding_file))
        natural_language_embedding_file='glove_s50'
        embedding_path= '{:}{:}.txt'.format(EMBEDDINGS_DIR, natural_language_embedding_file)        
        word2vec  = KeyedVectors.load_word2vec_format(embedding_path, unicode_errors="ignore")          
    finally:
        if verbose:
            print('Fetching word2vec... done')  

    return word2vec

def fetch_corpus_exceptions(corpus_exception_file, verbose=True):
    '''
        Returns a list of supported feature names

        args:
            corpus_exception_file

            ner_tag

        returns:
            features : list<str>  with the features names
    '''
    if verbose:
        print('Fetching {:}...'.format(corpus_exception_file))
    
    corpus_exceptions_path = '{:}{:}'.format(CORPUS_EXCEPTIONS_DIR, corpus_exception_file)
    df = pd.read_csv(corpus_exceptions_path, sep='\t')
    if verbose:
        print('Fetching {:}...done'.format(corpus_exception_file))  

    return set(df['TOKEN'].values)



def preprocess(lexicon, word2vec, verbose=True):  
    '''
        1. for NER entities within exception file
             replace by the tag organization, person, location
        2. for smaller than 5 tokens replace by one hot encoding 
        3. include time i.e 20h30, 9h in number embeddings '0'
        4. include ordinals 2º 2ª in number embeddings '0'
        5. include tel._38-4048 in numeber embeddings '0'

    New Word embedding size = embedding size + one-hot enconding of 2   
    '''
    # define outputs
    total_words = len(lexicon)
    lexicon2token = dict(zip(lexicon, ['unk']*total_words))

    # fetch exceptions list
    pers = fetch_corpus_exceptions('corpus-word-missing-pers.txt', verbose=verbose)
    locs = fetch_corpus_exceptions('corpus-word-missing-locs.txt', verbose=verbose)
    orgs = fetch_corpus_exceptions('corpus-word-missing-orgs.txt', verbose=verbose)


    #define regex
    re_punctuation = re.compile(r'[{:}]'.format(string.punctuation), re.UNICODE)
    re_number = re.compile(r'^\d+$')
    re_tel = re.compile(r'^tel\._')
    re_time = re.compile(r'^\d{1,2}h\d{0,2}$')
    re_ordinals = re.compile(r'º|ª')

    for word in list(lexicon):
        # some hiffenized words belong to embeddings
        # ex: super-homem, fim-de-semana, pré-qualificar, caça-níqueis
        token = word.lower() 
        if token in word2vec:                   
            lexicon2token[word]= token
        else:
            # if word in ['Rede_Globo', 'Hong_Kong', 'Banco_Central']:
            token = re_tel.sub('', token)
            token = re_ordinals.sub('', token)
            token = re_punctuation.sub('', token)

            token = re_time.sub('0', token)
            token = re_number.sub('0', token)

            if token in word2vec:
                lexicon2token[word]= token.lower()
            else:
                if word in pers:
                    lexicon2token[word] = 'pessoa'
                else:
                    if word in orgs:
                        lexicon2token[word] = 'organização'
                    else:
                        if word in locs:
                            lexicon2token[word] = 'local'

    total_tokens = len([val for val in lexicon2token.values() if not val in ('unk')])
    if verbose:
        print('Preprocess finished. Found {:} of {:} words, missing {:.2f}%'.format(total_tokens,
            total_words, 100*float(total_words-total_tokens)/ total_words))                

    return lexicon2token

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

    if ds_type  == 'train':
        return lb, ub
    else:
        lb += ub
        ub += size_dict['valid']
        if ds_type  == 'valid':
            return lb, ub
        elif ds_type  == 'test':
            lb += ub
            ub += size_dict['test']
        return lb, ub


def snapshot_hparam_string(embeddings='glo50', target_label='T',
                      is_batch=True, learning_rate=5 * 1e-3,
                      version='1.0',hidden_layers=[16] * 4, **kwargs):
    '''Makes a nested directory to record model's data

    Stores the parameters of a given experiment within a nested
    directory structure -- the inner directory will be a timestamp

    Keyword Arguments:
        embeddings {str} -- Word embeddings mneumonic (default: {'glo50'})
                `glo`, `wan` , `wrd` for GloVe, Wang2Vec and Word2Vec.
        target_label {str} -- Target label (default: {'T'})
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

            if key == 'embeddings':
                param_list[1] = value

            if key == 'hidden_layers':
                hidden_list_ = [str(s) for s in value]
                param_list[2] = 'x'.join(hidden_list_)
                param_list[2] = 'hs_{:}'.format(param_list[2])

            if key == 'ctx_p':
                param_list[3] = 'ctxp_{:d}'.format(value)

            if key == 'target_label':
                param_list[4] = value

            if key == 'is_batch':
                param_list[5] = 'batch' if value else 'kfold'

            if key == 'learning_rate':
                param_list[6] = 'lr_{:.2e}'.format(value)

    snapshot_dir = ''
    for param_ in param_list:
        snapshot_dir += '/{}'.format(param_)

    snapshot_dir += '/'

    return snapshot_dir


def snapshot_persist(target_dir, input_labels=None, target_label=None,
                    hidden_layers=None, embeddings=None,
                    epochs=None, lr=None, batch_size=None,
                    kfold=None, version='1.0', **kwargs):
    '''Saves the parameters for a given model 
    
    Creates a directory it one doesnot exist
    Saves the contents for the parameters in a timestamp dir

    Arguments:
        target_dir {[type]} -- [description]
        **kwargs {[type]} -- [description]

    Keyword Arguments:
        input_labels {[type]} -- [description] (default: {None})
        target_label {[type]} -- [description] (default: {None})
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

    target_path = '{:}params.txt'.format(target_dir)
    with open(target_path, mode='w') as f:
        if input_labels is not None:
            line_ = 'input_labels:\t'
            line_ += ','.join(input_labels)
            line_ += '\n'
            f.write(line_)

        if target_label is not None:
            line_ = 'target_label:\t'
            line_ += ','.join(target_label)
            line_ += '\n'
            f.write(line_)

        if hidden_layers is not None:
            line_ = 'hidden_layers:\t'
            line_ += 'x'.join([str(h) for h in hidden_layers])
            line_ += '\n'
            f.write(line_)

        if embeddings is not None:
            line_ = 'embeddings:\t'
            line_ += embeddings
            line_ += '\n'
            f.write(line_)

        if epochs is not None:
            line_ = 'epochs:\t'
            line_ += '{:d}'.format(epochs)
            line_ += '\n'
            f.write(line_)

        if lr is not None:
            line_ = 'learning_rate:\t'
            line_ += '{:.2e}'.format(lr)
            line_ += '\n'
            f.write(line_)

        if batch_size is not None:
            line_ = 'batch_size:\t'
            line_ += '{:d}'.format(batch_size)
            line_ += '\n'
            f.write(line_)

        if kfold is not None:
            line_ = 'kfold:\t'
            line_ += str(True)
            line_ += '\n'
            f.write(line_)

        if version is not None:
            line_ = 'version:\t'
            line_ += version
            line_ += '\n'
            f.write(line_)

    return target_dir
