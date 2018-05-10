'''
    Author: Guilherme Varela

    Performs dataset build according to refs/1421593_2016_completo
    1. Merges PropBankBr_v1.1_Const.conll.txt and PropBankBr_v1.1_Dep.conll.txt as specified on 1421593_2016_completo
    2. Parses new merged dataset into train (development, validation) and test, so it can be benchmarked by conll scripts 


'''
import sys 
import random 
import pandas as pd 
import numpy as np 
import re
import os.path
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
PROPBANKBR_PATH='../datasets/txts/conll/'
# PROPBANKBR_PATH='../datasets/conll/'
TARGET_PATH='../datasets/csvs/'
# PROPBANKBR_PATH='datasets/conll/'
# TARGET_PATH='datasets/csvs/'

#MAPS the filename and output fields to be harvested
CONST_HEADER=[
    'ID','FORM','LEMMA','GPOS','MORF', 'IGN1', 'IGN2', 
    'CTREE','IGN3', 'PRED','ARG0','ARG1','ARG2','ARG3',
    'ARG4','ARG5','ARG6'
]
DEP_HEADER=[
    'ID','FORM','LEMMA','GPOS','MORF', 'DTREE', 'FUNC', 
    'IGN1', 'IS_PRED', 'PRED','ARG0','ARG1','ARG2','ARG3',
    'ARG4','ARG5','ARG6'
]

MAPPER= {
    'CONST': { 
        'filename': 'PropBankBr_v1.1_Const.conll.txt',
        'mappings': {
            'ID':0,
            'FORM':1,
            'LEMMA':2,
            'GPOS':3,
            'MORF':4,
            'CTREE':7,
            'PRED':9,
            'ARG0':10,
            'ARG1':11,
            'ARG2':12,
            'ARG3':13,
            'ARG4':14,
            'ARG5':15,
            'ARG6':16,
        }
    }, 
    'DEP': { 
        'filename': 'PropBankBr_v1.1_Dep.conll.txt',
        'mappings': {
            'ID':0,
            'FORM':1,
            'LEMMA':2,
            'GPOS':3,
            'MORF':4,
            'DTREE':5,
            'FUNC':6,
            'IS_PRED':7,
            'PRED':8,
            'ARG0':9,
            'ARG1':10,
            'ARG2':11,
            'ARG3':12,
            'ARG4':13,
            'ARG5':14,
            'ARG6':15,
        }
    }
}

def propbankbr_lazyload(dataset_name='zhou'):
    dataset_path= TARGET_PATH + '/{}.csv'.format(dataset_name)  
    if os.path.isfile(dataset_path):
        df= pd.read_csv(dataset_path)       
    else:
        df= propbankbr_parser2()
        propbankbr_persist(df, split=True, dataset_name=dataset_name)         
    return df 
        
def propbankbr_persist(df, split=True, dataset_name='zhou'):
    df.to_csv('{}{}.csv'.format(TARGET_PATH ,dataset_name))
    if split: 
        dftrain, dfvalid, dftest=propbankbr_split(df)
        dftrain.to_csv( TARGET_PATH + '/{}_train.csv'.format(dataset_name))
        dfvalid.to_csv( TARGET_PATH + '/{}_valid.csv'.format(dataset_name))
        dftest.to_csv(  TARGET_PATH + '/{}_test.csv'.format(dataset_name))
    


def propbankbr_split(df, testN=263, validN=569):
    '''
        Splits propositions into test & validation following convetions set by refs/1421593_2016_completo
            |development data|= trainN + validationN 
            |test data|= testN

    ''' 
    P = max(df['P']) # gets the preposition
    Stest = min(df.loc[df['P']> P-testN,'S']) # from proposition gets the sentence  
    dftest= df[df['S']>=Stest]

    Svalid = min(df.loc[df['P']> P-(testN+validN),'S']) # from proposition gets the sentence    
    dfvalid= df[((df['S']>=Svalid) & (df['S']<Stest))]

    dftrain= df[df['S']<Svalid]
    return dftrain, dfvalid, dftest


def propbankbr_parser():
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
    df_dep2 = df_dep[['FUNC', 'DTREE', 'IS_PRED', 'S', 'P', 'P_S' ]]
    usecols = ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF',
        'DTREE', 'FUNC', 'CTREE', 'IS_PRED', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 
        'ARG5', 'ARG6'
    ]

    df= pd.concat((df_const, df_dep2), axis=1)
    df= df[usecols] 
    df= df.applymap(trim)

    return df

def propbankbr_parser2():
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
    # df_dep2 = df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
    df_const2 = df_const['CTREE']
    usecols = ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF',
        'DTREE', 'FUNC', 'CTREE', 'IS_PRED', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
        'ARG5', 'ARG6'
    ]

    df = pd.concat((df_dep, df_const2), axis=1)
    df = df[usecols]
    df = df.applymap(trim)
    df['GPOS'] = df['GPOS'].str.upper()

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
    filename=PROPBANKBR_PATH + MAPPER['DEP']['filename']
    mappings=MAPPER['DEP']['mappings']
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
                    
                    if (key=='IS_PRED'):
                        val = 1 if val=='Y' else 0

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



def propbankbr_y2arg2(P, ID, PRED, IS_PRED, Dtree, Y):
    # finds predicate time 
    root_d = {P[i]:i for i, pred in enumerate(PRED) if pred != '-'}
    lb = 0
    ub = 0
    prev_prop = -1
    prev_time = -1
    process = False
    last_ancestor = None
    ARG = []
    for t, proposition in enumerate(P):
        if prev_prop < proposition:
            if prev_prop > 0:
                lb = ub
                ub = prev_time + 1  # ub must be inclusive
                process = True
                last_ancestor = None

        if process:
            G, root = _dtree_build(Dtree, ID, IS_PRED, lb, ub)
            subroot = root_d[prev_prop]
            ARG_PRED = _dtree_process(G, subroot, lb, ub)
            ARG += ARG_PRED
            process = False

        prev_prop = proposition
        prev_time = t

    lb = ub
    ub = prev_time + 1  # ub must be inclusive
    last_ancestor = None
    G, root = _dtree_build(Dtree, ID, IS_PRED, lb, ub)
    subroot = root_d[prev_prop]
    isopen = False
    ARG_PRED = _dtree_process(G, subroot, lb, ub)
    ARG += ARG_PRED
    return ARG


def _dtree_process(G, predicate_node, lb, ub):
    ARG = []
    last_ancestor = None
    isopen = False
    for i in range(lb, ub):
        ancestor = _dtree_find_ancestor(G, predicate_node, i)
        if i == predicate_node: # verb found
            if last_ancestor is not None:
                if isopen:
                    ARG[-1] += ')'
                    isopen = False
            arg = '(V*)'
        else:
            if ancestor is None:
                if isopen:
                    ARG[-1] += ')'
                    isopen = False
                arg = '*'
            else:
                if ancestor != last_ancestor:
                    if isopen:
                        ARG[-1] += ')'
                    arg = '({:}*'.format(Y[ancestor]) if Y[ancestor] != '-' else '*'
                    isopen = Y[ancestor] != '-'
                else:
                    arg = '*'
        ARG.append(arg)
        last_ancestor = ancestor

    if isopen:
        ARG[-2] += ')'
        isopen = False
    return ARG


def _dtree_find_ancestor(G, predicate_node, current_node):
    '''
        Returns common ancestor node
        Each node follows the rule:
        1 - current_node == predicate_node --> return current_node
        2 - current_node in neightbors(predicate_node) --> return current_node
        3 - current_node in  grand_children(predicate_node) -->  return first_child
        4 - current_node is not in sublist --> return None

        args:
            G .: Dtree 
            predicate_node .:
            current_node .:

        returns:
            ancestor_node .:
    '''
    q = deque(list())
    _, ancestor_node = _dtree_dfs(G, predicate_node, current_node, q)
    _dtree_refresh(G)
    return ancestor_node


def _dtree_dfs(G, u, i, q):
    '''
        Returns searches for i in subtree rooted at u
        args:
            G .: networkx.Graph Dtree
            u .: int ancestor node
            i .: int node
            q .: deque

        returns:
            ancestor .: int or None
                if i is in subtree rooted at first u then
                ancestor == root if u == i 
                ancestor == min(neighbors(root), i)
    '''
    if i == 76:
        print('u:{:} --> i:{:}'.format(ID[u], ID[i]))
    G.nodes[u]['discovered'] = True
    q.append(u)

    # current node u is target node i
    if i == u:
        try:
            # pop all ancestors that are predicate
            # in order to find which is the root token
            while q and G.node[q[0]]['predicate']:
                ancestor = q.popleft()
            ancestor = q.popleft()
        except IndexError:
            pass
        return False, ancestor # target tag is equal to the first children
    else:
        # keep looking
        for v in G.neighbors(u):
            if not G.node[v]['discovered']:
                search, ancestor = _dtree_dfs(G, v, i, q)
                if not search:
                    return False, ancestor


    q.pop()
    return True, None


def _dtree_refresh(G):
    for u in G:
        G.nodes[u]['discovered'] = False


def _dtree_build(Dtree, ID, IS_PRED, lb, ub):
    G = nx.Graph()
    root = None
    for i in range(lb, ub):
        G.add_node(i, discovered=False, predicate=bool(IS_PRED[i]))

    for i in range(lb, ub):
        v = Dtree[i]
        u = ID[i]
        if v == 0:
            root = i
        else:
            G.add_edge(i, (v - u) + i)

    return G, root


def propbankbr_arg2t(propositions, arguments):
    '''
        Converts default argument 0 into argument 1  format for easier softmax

    '''
    prev_tag=''
    prev_prop=-1
    new_tags=[]
    for prop, tag in zip(propositions, arguments):
        if prev_prop == prop: 
            if (tag in ['*']): # Either repeat or nothing
                if (')' in prev_tag): 
                    new_tag='*'
                else:   
                    new_tag=new_tags[-1] # repeat last
            else:
                if  (')' in prev_tag): #last tag is closed, refresh                  
                    new_tag=re.sub(r'\(|\)|\*|','',tag)                 
                else:
                    if prev_tag != tag and tag != '*)':
                        new_tag=re.sub(r'\(|\)|\*|','',tag)                 
                    else:
                        new_tag=new_tags[-1]            
        else: 
            if (tag in ['*']):
                new_tag='*'
            else:
                new_tag=re.sub(r'\(|\)|\*|','',tag)

        prev_tag= tag   
        prev_prop= prop     
        new_tags.append(new_tag)
    return new_tags


def propbankbr_arg2r(arguments):
    '''
        Converts arguments to root_arguments .:
        Only the root of semantic chuck carries the tag.

    '''
    new_tags = []
    for tag in arguments:
        if (tag in ('*')):
            new_tag = '*'
        elif (tag in ('*)')):
            new_tag = ')'
        else:
            new_tag = re.sub(r'\(|\)|\*|', '', tag)

        new_tags.append(new_tag)
    return new_tags


def propbankbr_r2arg(forms, propositions, arguments):
    '''
        Root arguments into golden standard arguments

        Only the root of semantic chuck carries the tag.

    '''
    isopen = False
    prev_tag = '*'
    prev_prop = -1
    new_tags = []
    triples = zip(forms, propositions, arguments)
    i = 0 
    for form, prop, tag in triples:
        if prop != prev_prop:
            prev_tag = '*'
            if isopen: # Close 
                if ')' not in new_tags:
                    new_tags[-1] += ')'
                    isopen = False

        if tag == ')':
            new_tag = '*)'
            isopen = False

        elif tag != '*' and tag != prev_tag:
            if prev_prop == prop and\
             (prev_tag not in ('*', ')') or (form == '.' and isopen)):
                if ')' not in new_tags[-1]:
                    new_tags[-1] += ')'
                    isopen = False

            if tag != '*':
                new_tag = '({:}*'.format(tag)
                isopen = True

        elif prev_prop == prop:
            if form == '.' and isopen:
                # Close previous
                if ')' not in new_tags[-1]:
                    new_tags[-1] = '*)'
                    isopen = False
            new_tag = '*'
        else:
            new_tag = '*'


        prev_tag = tag if tag != '*' else prev_tag
        prev_prop = prop
        new_tags.append(new_tag)
        i += 1
    if isopen:
        new_tags[-1] += ')'
        isopen = False

    return new_tags


def get_signature(mappings): 
    return {k:[] for k in mappings}


def trim(val):
    if isinstance(val, str):
        return val.strip()
    return val

if __name__== '__main__':
    

    # Test propbank parsing
    # df = propbankbr_parser()  # --> needs update
    # propbankbr_persist(df, dataset_name='gssynth')
    # print('Parsing propbank')
    # df = propbankbr_parser2(ctx_p_size=3)
    # df_train, df_valid, df_test =propbankbr_split(df)

    #testing new arguments
    # dfgs = pd.read_csv('datasets/csvs/gs.csv', index_col=0, encoding='utf-8')
    # propositions = dfgs['P'].values
    # forms = dfgs['FORM'].values
    # arguments = dfgs['ARG'].values
    # testing Y2ARG
    # P = [0] * 9
    # ID = list(range(1, 10))
    # PRED = ['-'] * 9
    # PRED[-3] = 'negar'
    # Dtree = [2, 7, 2, 5, 2, 5, 0, 7, 7]
    # Y = ['-'] * 9
    # Y[1] = 'A0'
    # Y[7] = 'A1'

    # ARG = propbankbr_y2arg2(P, ID, PRED, Dtree, Y)
    # print(ARG)

    # P = [1] * 33
    # ID = list(range(1, 34))
    # PRED = ['-'] * 33
    # PRED[4] = 'revelar'
    # Dtree = [5, 5, 2, 3, 0, 7, 5, 7, 7,
    #          25, 12, 10, 12, 25, 17, 17, 25,
    #          17, 20, 17, 17, 17, 24, 22, 7,
    #          27, 25, 25, 28, 31, 29, 31, 5]
    # Y = ['-'] * 33
    # Y[1] = 'A0'
    # Y[6] = 'A1'

    # P += [2] * 33
    # ID += list(range(1, 34))
    # PRED += ['recusar' if i == 9 else '-' for i in range(33)] 
    # PRED[32 + 9] = 'revelar'
    # Dtree += [5, 5, 2, 3, 0, 7, 5, 7, 7,
    #          25, 12, 10, 12, 25, 17, 17, 25,
    #          17, 20, 17, 17, 17, 24, 22, 7,
    #          27, 25, 25, 28, 31, 29, 31, 5]
    # Y += ['-'] * 32
    # Y[32 + 12] = 'A1'
    # Y += ['A1' if i == 11 else '-' for i in range(33)] 
    # ARG = propbankbr_y2arg2(P, ID, PRED, Dtree, Y)
    # import code; code.interact(local=dict(globals(), **locals()))
    
    # dfsynth = pd.read_csv('datasets/csvs/gs.csv', index_col=0, encoding='utf-8', sep=',')
    dfdtree = pd.read_csv('datasets/csvs/gsdtree.csv', index_col=0, encoding='utf-8', sep=',')

    # propositions = list(dfsynth['P'].values)
    # arguments = list(dfsynth['ARG'].values)

    Y = list(dfdtree['ARG'].values)
    ID = list(dfdtree['ID'].values)
    P = list(dfdtree['P'].values)
    PRED = list(dfdtree['PRED'].values)
    IS_PRED = list(dfdtree['IS_PRED'].values)
    Dtree = list(dfdtree['DTREE'].values)

    
    # arg2t = propbankbr_arg2t(propositions, arguments)
    # t2arg = propbankbr_t2arg(propositions, arg2t)

    ARG = propbankbr_y2arg2(P[:132], ID[:132], PRED[:132], IS_PRED[:132], Dtree[:132], Y[:132])
    import code; code.interact(local=dict(globals(), **locals()))
    # y2arg = propbankbr_y2arg2(P, ID, PRED, IS_PRED, Dtree, Y)
    

    # arg2r = propbankbr_arg2r(arguments)
    # r2arg = propbankbr_r2arg(forms, propositions, arg2r)
    # with open('test_arguments.csv', mode='w') as f:
    #     f.write(',ARG2T,T2ARG,Y2ARG\n')
    #     for i, arg in enumerate(arguments):
    #         line = '{:},{:},{:},{:},{:}\n'.format(i, arg, arg2t[i], t2arg[i],  y2arg[i])
    #         f.write(line)
