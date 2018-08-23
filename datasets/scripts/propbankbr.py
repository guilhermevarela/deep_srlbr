'''
    Author: Guilherme Varela

    Performs dataset build according to refs/1421593_2016_completo
    1. Merges PropBankBr_v1.1_Const.conll.txt and PropBankBr_v1.1_Dep.conll.txt as specified on 1421593_2016_completo
    2. Parses new merged dataset into train (development, validation) and test, so it can be benchmarked by conll scripts 
    
    Update CoNLL 2004 Shared Task 

'''
import sys 
import random 
import pandas as pd 
import numpy as np 
import re
import os.path
#import networkx as nx
#import matplotlib.pyplot as plt
from collections import defaultdict, deque

PROPBANKBR_PATH = '../datasets/txts/conll/'
# PROPBANKBR_PATH = 'datasets/txts/conll/'

TARGET_PATH = '../datasets/csvs/'

# MAPS the filename and output fields to be harvested

CONST_HEADER = [
    'ID', 'FORM', 'LEMMA', 'GPOS', 'MORF', 'IGN1', 'IGN2',
    'CTREE', 'IGN3', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3',
    'ARG4', 'ARG5', 'ARG6'
]
DEP_HEADER = [
    'ID', 'FORM', 'LEMMA', 'GPOS', 'MORF', 'DTREE', 'FUNC',
    'IGN1', 'IS_PRED', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3',
    'ARG4', 'ARG5', 'ARG6'
]

HEADER = [
    'ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'GPOS', 'MORF', 'CTREE', 'PRED',
    'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARG6'
]
MAPPER_V1_0 = {
    'filename': ['PropBankBr_v1.0_Develop.conll.txt',
                 'PropBankBr_v1.0_Test.conll.txt'],
    'mappings': {
        'ID': 0,
        'FORM': 1,
        'LEMMA': 2,
        'GPOS': 3,
        'MORF': 4,
        'CTREE': 7,
        'PRED': 8,
        'ARG0': 9,
        'ARG1': 10,
        'ARG2': 11,
        'ARG3': 12,
        'ARG4': 13,
        'ARG5': 14,
        'ARG6': 15,
    }
}
MAPPER_V1_1 = {
    'CONST': {
        'filename': 'PropBankBr_v1.1_Const.conll.txt',
        'mappings': {
            'ID': 0,
            'FORM': 1,
            'LEMMA': 2,
            'GPOS': 3,
            'MORF': 4,
            'CTREE': 7,
            'PRED': 9,
            'ARG0': 10,
            'ARG1': 11,
            'ARG2': 12,
            'ARG3': 13,
            'ARG4': 14,
            'ARG5': 15,
            'ARG6': 16,
        }
    },
    'DEP': {
        'filename': 'PropBankBr_v1.1_Dep.conll.txt',
        'mappings': {
            'ID': 0,
            'FORM': 1,
            'LEMMA': 2,
            'GPOS': 3,
            'MORF': 4,
            'DTREE': 5,
            'FUNC': 6,
            'PRED': 8,
            'ARG0': 9,
            'ARG1': 10,
            'ARG2': 11,
            'ARG3': 12,
            'ARG4': 13,
            'ARG5': 14,
            'ARG6': 15,
        }
    }
}


def propbankbr_split(df, testN=263, validN=569):
    '''
        Splits propositions into test & validation following convetions set by refs/1421593_2016_completo
            |development data|= trainN + validationN 
            |test data|= testN

    '''
    P = max(df['P']) # gets the preposition
    Stest = min(df.loc[df['P']> P-testN,'S']) # from proposition gets the sentence  
    dftest = df[df['S']>=Stest]

    Svalid = min(df.loc[df['P']> P-(testN+validN),'S']) # from proposition gets the sentence    
    dfvalid = df[((df['S']>=Svalid) & (df['S']<Stest))]

    dftrain = df[df['S']<Svalid]
    return dftrain, dfvalid, dftest


def propbankbr_synthactic_1_1():
    '''
    Users arguments as of CONLL 2005 (FLATTENED ARGUMENT) <--> SYNTHATIC TREE

    'ID'    : Contador de tokens que inicia em 1 para cada nova proposição
    'FORM'  : Forma da palavra ou sinal de pontuação
    'LEMMA' : Lema gold-standard da FORM 
    'GPOS'  : Etiqueta part-of-speech gold-standard
    'MORF'  : Atributos morfológicos  gold-standard
    'DTREE' : Árvore Sintagmática gold-standard completa    
    'FUNC'  : Função Sintática do token gold-standard para com seu regente na árvore de dependência
    'CTREE' : Árvore Sintagmática gold-standard completa
    'PRED'  : Predicatos semânticos na proposição
    'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    '''
    df_const = _const_read()
    df_dep = _dep_read()

    # preprocess
    df_dep2 = df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
    usecols = ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF',
        'DTREE', 'FUNC', 'CTREE', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 
        'ARG5', 'ARG6'
    ]

    df= pd.concat((df_const, df_dep2), axis=1)
    df= df[usecols] 
    df= df.applymap(trim)

    return df

def propbankbr_dependency_1_1():
    '''
    Users arguments as of CONLL 2005 (ONLY THE ARGUMENT SETTING AT THE ROOT OF THE TREE) <--> DTREE

    'ID'    : Contador de tokens que inicia em 1 para cada nova proposição
    'FORM'  : Forma da palavra ou sinal de pontuação
    'LEMMA' : Lema gold-standard da FORM 
    'GPOS'  : Etiqueta part-of-speech gold-standard
    'MORF'  : Atributos morfológicos  gold-standard
    'DTREE' : Árvore Sintagmática gold-standard completa    
    'FUNC'  : Função Sintática do token gold-standard para com seu regente na árvore de dependência
    'CTREE' : Árvore Sintagmática gold-standard completa
    'PRED'  : Predicatos semânticos na proposição
    'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    '''
    df_const = _const_read()
    df_dep = _dep_read()

    # preprocess
    df_const2 = df_const['CTREE']
    usecols = ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF',
        'DTREE', 'FUNC', 'CTREE', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
        'ARG5', 'ARG6'
    ]

    df = pd.concat((df_dep, df_const2), axis=1)
    df = df[usecols]
    df = df.applymap(trim)
    df['GPOS'] = df['GPOS'].str.upper()

    return df

def propbankbr_parser_1_0():
    '''Parses the file creating a dataset

    PropBankBr parser

    Returns:
        df {pd.DataFrame} -- pandas dataframe
    '''
    rows_list = []
    # Computes the sentences
    sentence_list = []
    sentence_count = 1

    # Computes if the predicate has been seen
    predicate_sentence_count = 0
    predicate_sentence_list = []

    predicate_count = 0
    predicate_list = []
    mapping_dict = MAPPER_V1_0['mappings']
    column_labels = list(MAPPER_V1_0['mappings'].keys())
    column_ids = list(MAPPER_V1_0['mappings'].values())

    for filename in MAPPER_V1_0['filename']:
        file_path = '{:}{:}'.format(PROPBANKBR_PATH, filename)
        with open(file_path, mode='r') as f:
            for line_ in f.readlines():
                data_ = re.sub('\n', '', line_).split('\t')
                if len(data_) > 1:
                    row_ = [
                             trim(data_[column_]) if column_ < len(data_) else None
                             for column_ in list(column_ids)]

                    predicate_sentence_count = \
                        predicate_sentence_count + 1\
                        if row_[column_labels.index('PRED')] != '-' else predicate_sentence_count

                    predicate_sentence_list.append(predicate_sentence_count)
                    predicate_list.append(predicate_count + predicate_sentence_count)
                    sentence_list.append(sentence_count)
                    rows_list.append(row_)
                else:
                    sentence_count += 1
                    predicate_count += predicate_sentence_count
                    predicate_sentence_count = 0 

    df = pd.DataFrame(data=rows_list, columns=column_labels)

    df['S'] = sentence_list
    df['P_S'] = predicate_sentence_list
    df['P'] = predicate_list
    df = df[HEADER]
    return df


def propbankbr_parser(version='1.1', synthactic=True):
    '''Parses file returning a pandas DataFrame

    handles multiple versions of a dataset

    Arguments:
        version {str} -- containing the db version
        synthactic {bool} -- if true returns the arguments
            of synthactic tree else dependency argumets (version 1.1 only)

    Returns:
         dataframe {pd.DataFrame}

    Raises:
        ValueError -- version must be either 1.0 or 1.1
    '''
    if version not in ('1.0','1.1',):
        raise ValueError('version {:} not supported')
    else:
        if version in ('1.1',):
            if synthactic:
                df = propbankbr_synthactic_1_1()
            else:
                df = propbankbr_dependency_1_1()
        else:
            df = propbankbr_parser_1_0()
    return df


def propbankbr_argument_stats(df):
    '''
        Removes '', NaN, '*', 'V' in order to account only for valid tags
    '''
    def argument_transform(val):
        if isinstance(val, str):
            val = re.sub(r'C\-|\(|\)|\*|\\n| +', '',val)            
            val = re.sub(r' ', '',val)          
        else: 
            val=''
        return val

    dfarg= df[['ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG6']]
    dfarg= dfarg.applymap(argument_transform)
    values= dfarg.values.flatten()
    values= list(filter(lambda a : a != '' and a != 'V', values))

    stats= {k:0 for k in set(values)}
    for val in list(values): 
        stats[val]+=1
    return stats

def _const_read():
    '''
        Reads the file 'PropBankBr_v1.1_Const.conll.txt' returning a pandas DataFrame
        the format is as follows
        'ID'    : Contador de tokens que inicia em 1 para cada nova proposição
        'FORM'  : Forma da palavra ou sinal de pontuação
        'LEMMA' : Lema gold-standard da FORM 
        'GPOS'  : Etiqueta part-of-speech gold-standard
        'MORF'  : Atributos morfológicos  gold-standard
        'CTREE' : Árvore Sintagmática gold-standard completa
        'PRED'  : Predicatos semânticos na proposição
        'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    '''

    
    filename= PROPBANKBR_PATH + 'PropBankBr_v1.1_Const.conll.txt'
    df = pd.read_csv(filename, sep='\t', header=None, index_col=False, names=CONST_HEADER, dtype=str)
    
    del df['IGN1'] 
    del df['IGN2'] 
    del df['IGN3'] 

    return df 


def _dep_read():
    '''
        Reads the file 'PropBankBr_v1.1_Dep.conll.txt' returning a pandas DataFrame
        the format is as follows, 
        'ID'    : Contador de tokens que inicia em 1 para cada nova proposição
        'FORM'  : Forma da palavra ou sinal de pontuação
        'LEMMA' : Lema gold-standard da FORM 
        'GPOS'  : Etiqueta part-of-speech gold-standard
        'MORF'  : Atributos morfológicos  gold-standard
        'DTREE' : Árvore Sintagmática gold-standard completa
        'FUNC'  : Função Sintática do token gold-standard para com seu regente na árvore de dependência
        'PRED'  : Predicatos semânticos na proposição
        'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
        'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
    '''
    #malformed file prevents effective use of pd.read_csv
    filename=PROPBANKBR_PATH + MAPPER_V1_1['DEP']['filename']
    mappings=MAPPER_V1_1['DEP']['mappings']
    mappings_inv= {v:k for k,v in mappings.items()}
    
    sentences=get_signature(mappings)
    sentence_count=[]
    s_count=1                                    # counter over the number of sentences
    p_count=0                    # counter over the number of predicates
    ps_count=0                                   # predicate per sentence
    proposition_count=[]    
    proposition_per_sentence_count=[]   

    M=max(mappings_inv.keys())   # max number of fields

    for line in open(filename):         
        end_of_sentence= (len(line)==1)
        if end_of_sentence: 
            s_count+=1
            ps_count=0


        if not(end_of_sentence):
            data= line.split(' ')                   
            data= filter(lambda s: s != '' and s !='\n', data)          
            values= list(data)

            key_max=0
            for keys_count, val in enumerate(values): 
                if keys_count in mappings_inv.keys():
                    key=mappings_inv[keys_count]    

                    if (key=='PRED'):
                        if (val != '-'):
                            p_count+=1
                            ps_count+=1 

                    if (key[:3]=='ARG'):                        
                        val= val.translate(str.maketrans('','','\n()*')) 
                        sentences[key].append(val)
                    else:
                        sentences[key].append(val)
                key_max=keys_count

            #   fills remaning absent/optional ARG arrays with None
            # garantees all arrays have the same number of elements
            for keys_count in range(key_max+1, M+1, 1): 
                key=mappings_inv[keys_count]    
                sentences[key].append(None)

            sentence_count.append(s_count)
            proposition_count.append(p_count)
            proposition_per_sentence_count.append(ps_count)
    
    
    sentences['S']= sentence_count # adds a new column with number of sentences
    sentences['P']= proposition_count # adds a new column with number of propositions
    sentences['P_S']=proposition_per_sentence_count # adds a new column with number of propositions per sentence

    df = pd.DataFrame.from_dict(sentences)              
    # garantee a friedlier ordering of the columns
    cols=['ID', 'S' , 'P', 'P_S'] + list(mappings.keys())[1:]
    df = df[cols]
    
    return df


def propbankbr_t2arg(propositions, arguments):
    isopen=False
    prev_tag=''
    prev_prop=-1
    new_tags=[]
    
    for prop, tag in zip(propositions, arguments):
        if prop != prev_prop:
            prev_tag=''
            if isopen: # Close 
                new_tags[-1]+= ')' 
                isopen=False
            
        if tag != prev_tag:         
            if prev_tag != '*' and prev_prop == prop:
                new_tags[-1]+= ')' 
                isopen=False

            if tag != '*':  
                new_tag= '({:}*'.format(tag)
                isopen=True
            else:
                new_tag='*'
        elif prev_prop == prop:
            new_tag= '*'


        prev_tag= tag   
        prev_prop= prop
        new_tags.append(new_tag)

    if isopen:
        new_tags[-1]+=')'       
        isopen=False

    return new_tags


def propbankbr_arg2t(propositions, arguments):
    '''
        Converts default argument 0 into argument 1  format for easier softmax

    '''
    prev_tag = ''
    prev_prop = -1
    new_tags = []
    for prop, tag in zip(propositions, arguments):
        if prev_prop == prop:
            if (tag in ['*']):  # Either repeat or nothing
                if (')' in prev_tag):
                    new_tag = '*'
                else:
                    new_tag = new_tags[-1]  # repeat last
            else:
                if (')' in prev_tag):  # last tag is closed, refresh
                    new_tag = re.sub(r'\(|\)|\*|', '', tag)
                else:
                    if prev_tag != tag and tag != '*)':
                        new_tag = re.sub(r'\(|\)|\*|', '', tag)
                    else:
                        new_tag = new_tags[-1]
        else:
            if (tag in ['*']):
                new_tag = '*'
            else:
                new_tag = re.sub(r'\(|\)|\*|', '', tag)

        prev_tag = tag
        prev_prop = prop
        new_tags.append(new_tag)
    return new_tags


def propbankbr_arg2iob(propositions, arguments):
    '''
        Converts FLAT TREE format to IOB format

        args:
            proposition  .: list<int>
            arguments    .: list<str>

        returns:
            iob          .: list<str>
    '''
    prev_tag = ''
    prev_prop = -1
    new_tags = []
    for prop, tag in zip(propositions, arguments):
        if prev_prop == prop:
            if (tag in ['*', '*)']):
                if ((new_tags[-1] in ['O']) or ('*)' in prev_tag)):
                    new_tag = 'O'
                else:
                    new_tag = 'I-{:}'.format(re.sub(r'I-|B-|', '', new_tags[-1]))
            else:
                new_tag = 'B-{:}'.format(re.sub(r'\(|\)|\*|', '', tag))
        else:
            if (tag in ['*']):
                new_tag = 'O'
            else:
                new_tag = 'B-{:}'.format(re.sub(r'\(|\)|\*|', '', tag))

        prev_tag = tag
        prev_prop = prop
        new_tags.append(new_tag)
    return new_tags


def propbankbr_iob2arg(propositions, arguments):
    '''
        Converts IOB format to FLAT TREE format

        args:
            proposition  .: list<int>
            arguments    .: list<str>

        returns:
            iob          .: list<str>
    '''
    prev_tag = 'O'
    prev_prop = -1
    new_tags = []
    cop = 0
    ccl = 0
    for prop, tag in zip(propositions, arguments):
        if prev_prop == prop:
            if tag[:2] in ('B-',):
                # if prev_tag[:2] in ('B-',) or prev_tag[2:] != tag[2:]:
                if prev_tag != 'O':
                    ccl += 1
                    new_tags[-1] += ')'
                new_tag = re.sub(r'I-|B-', '(', tag) + '*'
                cop += 1
            elif tag[:2] in ('I-',):
                if prev_tag[2:] != tag[2:]:
                    if prev_tag != 'O':
                        ccl += 1
                        new_tags[-1] += ')'
                    new_tag = re.sub(r'I-|B-', '(', tag) + '*'
                    cop += 1
                else:
                    new_tag = '*'
            elif tag == 'O':
                if prev_tag != 'O':
                    ccl += 1
                    new_tags[-1] += ')'
                new_tag = '*'
            else:
                raise ValueError('Unknown {:}', tag)

        else:
            if cop > ccl:
                new_tags[-1] += ')'
            cop = 0
            ccl = 0
            if tag == 'O':
                new_tag = '*'
            elif tag[:2] in ('I-', 'B-') :
                new_tag = re.sub(r'I-|B-', '(', tag) + '*'
                cop += 1
            else:
                raise ValueError('Unknown {:}', tag)



        prev_tag = tag
        prev_prop = prop
        new_tags.append(new_tag)
    if (prev_tag[0] != 'O' and cop > ccl):
        new_tags[-1] += ')'
    return new_tags


def propbankbr_arg2se(propositions, arguments):
    '''Converts CoNLL 2005 Shared Task tags to CoNLL 2004 Shared Task

    Flat tree to start and end format ex: 

            CoNLL 2004 ST       CoNLL 2005 ST
   The         (A0*    (A0*       (A0*    (A0*
   $              *       *          *       *
   1.4            *       *          *       *
   billion        *       *          *       *
   robot          *       *          *       *
   spacecraft     *A0)    *A0)       *)      *)
   faces        (V*V)     *        (V*)      *
   a           (A1*       *       (A1*       *
   six-year       *       *          *       *
   journey        *       *          *       *
   to             *       *          *       *
   explore        *     (V*V)        *     (V*)
   Jupiter        *    (A1*          *    (A1*
   and            *       *          *       *
   its            *       *          *       *
   16             *       *          *       *
   known          *       *          *       *
   moons          *A1)    *A1)       *)      *)
   .              *       *          *       *

    Arguments:
        propositions {list{int}} -- integer
        arguments {list{str}} -- target
    '''
    se_list = []
    last_prop = -1
    matcher_open = re.compile(r'\(([A-Z0-9\-]*?)\*$')
    matcher_enclosed = re.compile(r'\(([A-Z0-9\-]*?)\*\)$')
    for prop, arg_tuple in zip(propositions, arguments):
        if last_prop != prop:
            open_arg_list = [''] * (len(arg_tuple) - 1)
            last_prop = prop
        arg_list = list(arg_tuple)
        se_i_list = arg_list[:1]
        # For each propositon
        for i_, arg_ in enumerate(arg_list[1:]):
            # open and close argument
            if arg_ == '*':
                se_arg_ = '*'
            else:
                matched = matcher_enclosed.match(arg_)
                if matched:

                    se_arg_ = matched.groups()[0]
                    se_arg_ = '({}*{})'.format(se_arg_, se_arg_)
                else:
                    matched = matcher_open.match(arg_)
                    if matched:
                        se_arg_ = arg_
                        open_arg_list[i_] = matched.groups()[0]
                    elif arg_ == '*)' and open_arg_list[i_] != '':
                        se_arg_ = '*{})'.format(open_arg_list[i_])
                    else:
                        msg_ = 'invalid argument {} on prop {}'.format(arg_, prop)
                        raise ValueError(msg_)
            se_i_list.append(se_arg_)
        se_list.append(tuple(se_i_list))
    return se_list

def get_signature(mappings): 
    return {k:[] for k in mappings}


def trim(val):
    if isinstance(val, str):
        return val.strip()
    return val

if __name__== '__main__':
    # dfgs = pd.read_csv('datasets/csvs/gs.csv', index_col=0, encoding='utf-8', sep=',')    

    # propositions = list(dfgs['P'].values)
    # arguments = list(dfgs['ARG'].values)

    # FORM = list(dfgs['FORM'].values)
    # ARG = list(dfgs['ARG'].values)
    # ID = list(dfgs['ID'].values)
    # P = list(dfgs['P'].values)
    # PRED = list(dfgs['PRED'].values)

    # Dtree = list(dfgs['DTREE'].values)
    # Ctree = list(dfgs['CTREE'].values)


    # arg2t = propbankbr_arg2t(propositions, arguments)
    # t2arg = propbankbr_t2arg(propositions, arg2t)

    # arg2iob = propbankbr_arg2iob(propositions, arguments)
    # iob2arg = propbankbr_iob2arg(propositions, arg2iob)
    # # import code; code.interact(local=dict(globals(), **locals()))
    # with open('test_arguments.csv', mode='w') as f:
    #     f.write('INDEX\tARG\tARG2T\tARG2IOB\tT2ARG\tIOB2ARG\n')
    #     for i, arg in enumerate(arguments):
    #         line = '{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n'.format(i, arg, arg2t[i], arg2iob[i], t2arg[i], iob2arg[i])
    #         f.write(line)
    df = propbankbr_parser_1_0()
    print(df.head())
