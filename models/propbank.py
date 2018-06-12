'''
Created on Mar 08, 2018
    @author: Varela
    
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
import sys
ROOT_DIR = '/'.join(sys.path[0].split('/')[:-1]) #UGLY PROBLEM FIXES TO LOCAL ROOT --> import configig
sys.path.append(ROOT_DIR)
sys.path.append('datasets/')

import config
import numpy as np
import pandas as pd
import pickle

import data_propbankbr as br
from models.utils import fetch_word2vec, fetch_corpus_exceptions, preprocess
from collections import OrderedDict

CSVS_DIR= './datasets/csvs/'


class _PropbankIterator(object):
    '''
    Iterates propositions

    ''' 
    def __init__(self, low, high, fnc):
        self.current= low 
        self.high= high
        self.fnc= fnc

    def __iter__(self):
        return self

    def __next__(self): 
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1, self.fnc(self.current-1)



class Propbank(object):
    '''
    Stores an indexed representation of the db and of the embeddings
    id= (propbank, embeddings_model)

    
    Translates propositions to columns providing both the 
        data itself and information about the data to machine learning experiments

    Provides iteration and columns in the formats ( index, text, embedded )



    ''' 
    lexicon = set([])
    lex2tok = {}
    tok2idx = {}
    idx2tok = {}
    embeddings = np.array([])
    embeddings_model = ''
    embeddings_sz = 0

    onehot = {}
    hotone = {}
    data = {}
    encodings = ('CAT', 'IDX', 'HOT', 'EMB')
    
    def __init__(self, db_name='db_pt', lexicon_columns=['LEMMA'], language_model='glove_s50', verbose=True):
        '''
            Returns new instance of propbank class - instantiates the data structures
            
            returns:
                propbank        .: object an instance of the Propbank class
        '''
        if (self.total_words() == 0):
            path =  '{:}{:}.csv'.format(CSVS_DIR, db_name, index_col=1)
            df = pd.read_csv(path)

            self.db_name = db_name
            self.lexicon_columns = lexicon_columns
            self.language_model = language_model
            self.embeddings_model, self.embeddings_sz = language_model.split('_s')
            self.embeddings_sz = int(self.embeddings_sz)

            for col in lexicon_columns:
                self.lexicon = self.lexicon.union(set(df[col].values))

            word2vec = fetch_word2vec(self.language_model)
            vocab_sz = self.total_words()

            # Preprocess
            if verbose:         
                print('Processing total lexicon is {:}'.format(self.total_words())) 
                    
            self.lex2tok = preprocess(list(self.lexicon), word2vec)
            self.tok2idx = {'unk':0}
            self.idx2tok = {0:'unk'}
            self.embeddings = np.zeros((vocab_sz, self.embeddings_sz), dtype=np.float32)
            self.embeddings[0] = word2vec['unk']

            tokens = set(self.lex2tok.values())
            idx = 1
            for token in list(tokens):
                if not(token in self.tok2idx):
                    self.tok2idx[token] = idx
                    self.idx2tok[idx] = token
                    self.embeddings[idx] = word2vec[token]
                    idx += 1

 
            self.columns = set(df.columns).intersection(set(config.SEQUENCE_FEATURES))
            self.columns = self.columns.union(set(['INDEX']))
            
            #ORDER of columns will be alphabetic this is important when 
            #columns become tensors
            for col in self.columns:
                #ALWAYS ADD INDEX in order to make merge possible after training
                if col in ['INDEX']: 
                    self.data[col]  = OrderedDict(zip(df.index, df.index))
                else: 
                    if col in config.META:
                        if config.META[col]  == 'hot':
                            keys= set(df[col].values)
                            idxs= list(range(len(keys)))
                            self.onehot[col]= OrderedDict(zip(keys, idxs))
                            self.hotone[col]= OrderedDict(zip(idxs, keys))
                        # for categorical variables 
                        if config.META[col] in ['hot']:
                            self.data[col] = OrderedDict(zip(
                                df[col].index, 
                                [self.onehot[col][val] 
                                    for val in df[col].values]
                            ))
                        elif config.META[col] in ['txt']:
                            self.data[col] = OrderedDict(zip(
                                df[col].index, 
                                [self.tok2idx[self.lex2tok[val]] 
                                    for val in df[col].values]
                            ))
                        else:
                            self.data[col] = OrderedDict(df[col].to_dict())
        else:
            raise  Exception('Lexicon and embeddings already defined')

    @classmethod
    def recover(cls, file_path):
        '''
        Returns a copy from the instanced saved @file_path
        args:
            file_path .: string a full path to a serialized dump of propbank object

        returns:
             object   .: propbank object instanced saved at file_path
        '''
        with open(file_path, 'rb') as f:
            propbank_instance= pickle.load(f)
        return propbank_instance

    def persist(self, file_dir, filename=''):
        '''
        Serializes this object in pickle format @ file_dir+filename
        args:
            file_dir    .: string representing the directory
            filename  .: string (optional) filename
    '''
        if not(filename):
            strcols = '_'.join( [col for col in self.lexicon_columns])
            filename = '{:}{:}_{:}_{:}.pickle'.format(file_dir, self.db_name, strcols, self.language_model)
        else:
            filename = '{:}{:}.pickle'.format(file_dir, filename)

        
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


    def decode(self, value, feature, apply_embeddings=False):
        '''
            Returns feature word from indexed representation
                this process may have losses since some on the lexicon are mapped to the same lemma
                (That applies to column LEMMA)

            args:
                value           .: int   representing a db record index
                feature     .: sring representing db column name

            returns:
                raw       .: string or int representing the value within original database
        '''
        result= value       
        if config.META[feature] in ['hot']:
            result= self.hotone[feature][value]

            if apply_embeddings:
                sz= self.size(feature)
                tmp= np.zeros((sz,),dtype=np.int32)
                tmp[result]=1
                result=tmp 

        elif config.META[feature] in ['txt']:
            result= self.idx2tok[value]

            if apply_embeddings:
                result= self.embeddings[result]
        return result


    def tensor2column(self, tensor_index, tensor_values, times, column):
        '''
            Converts a zero padded tensor to a dict
        
            Tensors must have the following shape [DATABASE_SIZE, MAX_TIME] with
                zeros if for t=0,....,MAX_TIME t>times[i] for i=0...DATABASE_SIZE 

        args:
            tensor_index  .: with db index

            tensor_values .: with db int values representations

            times  .: list<int> [DATABASE_SIZE] holding the times for each proposition

            column .: str           db column name

        returns:
            column .: dict<int, str> keys in db_index, values in columns or values in targets
        '''
        if not(column) in self.data:
            buff= '{:} must be in {:}'.format(column, self.data)
            raise KeyError(buff)

        index=[item  for i, sublist in enumerate(tensor_index.tolist()) 
            for j, item in enumerate(sublist) if j < times[i]]

        values=[self.decode(item, column, False)  for i, sublist in enumerate(tensor_values.tolist()) 
            for j, item in enumerate(sublist) if j < times[i]]

        return dict(zip(index, values))

    def t2arg(self, T):
        '''
            Converts column T into ARG

            args:
                T .: dict<int, str> keys in db_index, values: prediction label

            args:
                ARG .: dict<int, str> keys in db_index, values in target label
        '''

        propositions= {idx: self.data['P'][idx] for idx in T}

        ARG= br.propbankbr_t2arg(propositions.values(), T.values()) 

        return dict(zip(T.keys(), ARG))

    def total_words(self):
        return len(self.lexicon)    

    def size(self, columns):   
        '''
            Returns the dimension of the columns 
            args:
                columns : str or list<str>  feature name or list of feature names

            returns:
                sz          .: int indicating the total feature size
        '''
        if isinstance(columns,str):
            sz= self.column_dimensions(columns)
        else:
            sz=0
            for col in columns:
                sz+=self.column_dimensions(col)
        return sz

    def sequence_obsexample(self, idx, columns, as_dict=False, decode=False):
        '''
            Produces a record or observation from a sequence

            args:
                idx                 .: int   representing a db record index
                columns      .: list<sring> representing db column name
                as_dict     .: bool  if 1 then return as dictionary, else list
                decode      .: bool  if 1 then turn to lemma of string, else leave as index

            returns:
                raw                 .: list<columns> or dict<columns> indicating int value of columns
        '''     
        if as_dict:
            if decode:
                return {col: self.decode(self.data[col][idx], col)
                    for col in columns}
            else:
                return {col: self.data[col][idx] for col in columns}
        else:
            if decode:
                return [self.decode(self.data[col][idx], col)
                    for col in columns]
            else:
                return [self.data[col][idx] for col in columns]
        
    def sequence_obsexample2(self, categories, columns, embeddify=False):
        '''
            Produces a sequence from categorical values 
                (an entry) from the dataset

            args:
                values   .: list<str>  rawtext from csv
                columns .: list<str>  with the columns names
                embeddify   .: bool if false will return categorical indexes
                                                 if  true   will apply embeddings or onehot

            returns:
                example     .: numerical<N> if embeddify=True then N = self.size(columns) 
                                                                    else N=len(categories)  
        '''     
        
        feat2sz= { col: self.size(col)
            for col in columns}       

        sz=sum(feat2sz.values())    
        if embeddify:
            result= np.zeros((1,sz),dtype=np.float32)
        else:
            result=[]

        i=0
        j=0
        for col, sz in feat2sz.items():
            if embeddify:
                result[j:j+sz]= float(self.decode( categories[i], col, apply_embeddings=embeddify))
            else:
                result += self.decode( categories[i], col, false)

            i+=1
            j=sz 
        
        return result

    def column(self, ds_type, feature, decode=False):
        '''
            Returns a feature from data
                this process may have losses since some on the lexicon are mapped to the same lemma
                (That applies to column LEMMA)

            args:
                ds_type         .: string dataset name in (['train', 'valid', 'test'])
                feature     .: string representing db column name
                decode      .: bool representing the better encoding

            returns:
                feature  .: dict<int, value>
                
        
        '''
        if not(ds_type in ['train', 'valid', 'test']):
            buff= 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
            raise ValueError(buff)
        else:       
            if ds_type in ['train']:
                lb=0
                ub=config.DATASET_TRAIN_SIZE 
            elif ds_type in ['valid']:
                lb=config.DATASET_TRAIN_SIZE 
                ub=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE
            else:
                lb=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE 
                ub=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE + config.DATASET_TEST_SIZE
            if decode:
                d={idx: self.decode(self.data[feature][idx],feature)                    
                    for idx, p in self.data['P'].items()
                        if p>lb and p< ub+1}
            else:
                #unzips columns
                d=  { idx: self.data[feature][idx]
                    for idx, p in self.data['P'].items()
                        if p>=lb and p< ub+1}
        return d

    def to_svm(
               self, svm_path='datasets/svms/',
               encoding='EMB', excludecols=['ARG'], target='T'):
        '''
            Dumps data into csv
        '''
        if encoding.upper() not in self.encodings:
            raise ValueError('dump_type must be in {:}'.format(('CAT', 'IDX', 'HOT', 'EMB')))

        if encoding.upper() not in ('EMB', 'HOT'):
            raise NotImplementedError('You must implement dump_type=={:}'.format(encoding.lower()))

        columns = [col
                   for col in self.columns
                   if col not in excludecols and not col == target]

        _name = '_'.join(self.lexicon_columns)
        _name = '{:}_{:}'.format(_name, self.language_model)
        for ds_type in ['test', 'valid', 'train']:
            name = '{:}_{:}.svm'.format(ds_type, _name)
            file_path = '{:}{:}/{:}'.format(svm_path, encoding.lower(), name)
            with open(file_path, mode='w') as f:
                for idx, d in self.iterator(ds_type):
                    i = 1
                    line = '{:} '.format(d[target])
                    for col in columns:
                        sz = self.column_dimensions(col, encoding)
                        if config.META[col] in ['txt']:
                            if encoding.upper() in ['EMB']:
                                line += ' '.join(
                                    ['{:}:{:.6f}'.format(i + j, x)
                                     for j, x in enumerate(self.embeddings[d[col]])]
                                ) + ' '
                            elif encoding.upper() in ['HOT']:
                                onehot = [0] * sz
                                onehot[d[col]] = 1
                                line += ' '.join(
                                    ['{:}:{:}'.format(i + j, onehot[j]) for j in range(sz)]
                                ) + ' '
                        elif config.META[col] in ['hot']:
                            onehot = [0] * sz
                            onehot[d[col]] = 1
                            line += ' '.join(
                                ['{:}:{:}'.format(i + j, onehot[j]) for j in range(sz)]
                            ) + ' '
                        elif config.META[col] in ['int']:
                            line += '{:}:{:}'.format(i, d[col]) + ' '
                        i += sz + 1

                    line = line[:-1] + '\n'
                    f.write(line)

    def to_svm2(self, svm_path='datasets/svms/', deep_columns=True):
        '''
            Select between linguistic or deep columns            
        '''
        excludecols = ['INDEX', 'S', 'P', 'P_S']
        if deep_columns:
            # exclude linguistic columns
            excludecols += ['GPOS', 'MORF', 'DTREE', 'FUNC', 'CTREE', 'PRED']
            encoding = 'EMB'
        else:
            # exclude deep learning columns
            excludecols += ['LEMMA', 'CTX_P-3', 'CTX_P-2', 'CTX_P-1', 'CTX_P+1', 'CTX_P+2', 'CTX_P+3', 'MARKER', 'PRED_1']
            encoding = 'HOT'

        self.to_svm(svm_path=svm_path, encoding=encoding, excludecols=excludecols)




    def iterator(self, ds_type, decode=False):
        if not(ds_type in ['train', 'valid', 'test']):
            buff= 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
            raise ValueError(buff)
        else:
            as_dict=True
            fn = lambda x : self.sequence_obsexample(x, self.columns, as_dict, decode)
            if ds_type in ['train']:
                lb=0
                ub=config.DATASET_TRAIN_SIZE 
            elif ds_type in ['valid']:
                lb=config.DATASET_TRAIN_SIZE 
                ub=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE
            else:
                lb=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE 
                ub=config.DATASET_TRAIN_SIZE + config.DATASET_VALID_SIZE + config.DATASET_TEST_SIZE


        low=-1
        high=-1     
        prev_idx=-1
        
        for idx, prop in self.column(ds_type,'P').items():
            if prop > lb and low==-1:
                low= idx 
            
            if prop > ub and high==-1:
                high= prev_idx
                break
            prev_idx=idx

        if high==-1:
            high=prev_idx

        return _PropbankIterator(low, high, fn)

    def column_dimensions(self, col, encoding='EMB'):
        '''
            Returns the dimension of the column 
            args:
            returns:
        '''

        if encoding.upper() in ['CAT', 'IDX']:
            return 1

        sz = 1
        if config.META[col] in ['txt']:
            if encoding in ['EMB']:
                sz = len(self.embeddings[0])
            elif encoding in ['HOT']:
                sz = len(self.lexicon)
            else:
                raise ValueError('column_dimensions: encoding=={:} not supported'.format(encoding))

        if config.META[col] in ['hot']:
            sz = len(self.onehot[col])

        return sz

    def columns_dimensions(self, encoding='EMB'):
        '''
            Returns the dimension of the column 
            args:
            returns:
        '''
        return {
            col: self.column_dimensions(col, encoding=encoding)
            for col in self.columns
        }


if __name__ == '__main__':
    PROP_DIR = '{:}/datasets/binaries/'.format(ROOT_DIR)
    PROP_PATH = '{:}/{:}'.format(PROP_DIR, 'db_pt_LEMMA_glove_s50.pickle')

    propbank = Propbank(language_model='glove_s50')
    # propbank.define(language_model='wang2vec_s100')
    # propbank.define(language_model='glove_s50')
    # propbank.persist(PROP_DIR)
    # propbank = Propbank.recover(PROP_PATH)
    propbank.to_svm2(deep_columns=False)

    # PRED_d = propbank.feature('valid', 'PRED', True) # text
    # M_R_d = propbank.feature('valid', 'MARKER', True) # numerical
    # ARG_d = propbank.feature('valid', 'ARG', True) # categorical
    # print('found ARGs>\n',set(ARG_d.values()))
    # import code; code.interact(local=dict(globals(), **locals()))     

