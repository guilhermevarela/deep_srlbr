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

from gensim.models import KeyedVectors
import pandas as pd

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
    re_ordinals = re.compile(r'º|ª|primeir[ao]|segund[ao]')

    for word in list(lexicon):
        # some hiffenized words belong to embeddings
        # ex: super-homem, fim-de-semana, pré-qualificar, caça-níqueis
        token = word.lower()
        if token in word2vec:
            lexicon2token[word] = token
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
                        else:
                            tokens_list = word.split('_')

                            if len(tokens_list) > 1:  # It's a composite token
                                tokens_list = [tok_.lower()
                                               if tok_.lower() in word2vec else 'unk'
                                               for tok_ in tokens_list]

                                known_list = [tok_ for tok_ in tokens_list if tok_ != 'unk']
                                if len(known_list) > 0:
                                    lexicon2token[word] = '_'.join(tokens_list)

    total_tokens = len([val for val in lexicon2token.values() if not val in ('unk')])
    if verbose:
        print('Preprocess finished. Found {:} of {:} words, missing {:.2f}%'.format(total_tokens,
            total_words, 100*float(total_words-total_tokens)/ total_words))                

    return lexicon2token    