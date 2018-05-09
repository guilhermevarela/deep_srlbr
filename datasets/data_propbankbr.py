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
    'IGN1', 'PRED','ARG0','ARG1','ARG2','ARG3',
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
    df_dep2 = df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
    usecols = ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF',
        'DTREE', 'FUNC', 'CTREE', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 
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
        'DTREE', 'FUNC', 'CTREE', 'PRED', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
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


def propbankbr_y2arg(P, ID, PRED, Dtree, Y):
    # Stores the PRED index for the verb
    root_d = {P[i]:ID[i] for i, pred in enumerate(PRED) if pred != '-'}
    ARG = []
    prev_p = -1
    isopen = False # flags if argument is open
    for i in range(len(Y)):
        if P[i] != prev_p:
            root = root_d[P[i]]
            unlabeled = deque([])
            levels = defaultdict(list)
            labels = {}

            if prev_p > 0 and i + 1 < len(Y):
                if unlabeled:
                    try:
                        while True:
                            unlabeled.pop()
                            ARG.append('*')
                    except IndexError:
                        pass
                if isopen:
                    ARG[-2] += ')'
                    ARG[-1] = '*'
                    isopen = False
        # if i == 32 + 12:
        #     import code; code.interact(local=dict(globals(), **locals()))
        if ID[i] != root:
            unlabeled.append(i)
            if Dtree[i] == root:
                levels[0].append(ID[i])
                labels[ID[i]] = Y[i]
                if isopen and Y[i] == '-':
                    # Changed arguments
                    ARG[-1] += ')'
                    isopen = False
                else:
                    isopen = True
                try:
                    j = 0
                    while True:
                        unlabeled.pop()
                        y = '({:}*'.format(Y[i]) if j == 0 and Y[i] != '-' else '*'
                        ARG.append(y)
                        j += 1
                except IndexError:
                    pass
            else:

                # belongs to one of the subtrees?
                for l, nodes in levels.items():
                    if Dtree[i] in nodes:
                        y = labels[Dtree[i]]
                        try:
                            while True:
                                unlabeled.pop()
                                ARG.append('*')
                        except IndexError:
                            levels[l + 1].append(ID[i])
                            labels[ID[i]] = y
                        finally:
                            break
        else:
            if isopen:
                ARG[-1] += ')'
                isopen = False
            try:
                while True:
                    unlabeled.pop()
                    ARG.append('*')
            except IndexError:
                ARG.append('(V*)')
        print(P[i], prev_p, isopen, ARG)
        prev_p = P[i]

    if unlabeled:
        try:
            while True:
                unlabeled.pop()
                ARG.append('*')
        except IndexError:
            pass

    if isopen:
        ARG[-2] += ')'
        ARG[-1] = '*'
    return ARG


def propbankbr_y2arg2(P, ID, PRED, Dtree, Y):
        # finds predicate time 
        root_d = {P[i]:ID[i] for i, pred in enumerate(PRED) if pred != '-'}
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
                G, root = _dtree_build(Dtree, ID, lb, ub)
                subroot = root_d[prev_prop]
                isopen = False
                for i in range(lb, ub):
                    subroot = root_d[prev_prop]
                    q = deque(list())
                    _, ancestor = _dtree_dfs(G, root, i, q)
                    print('i:{:}\tID[i]:{:}\tancestor:{:}'.format(i, ID[i], ancestor))
                    if ID[i] == subroot: # verb found
                        if last_ancestor is not None:
                            ARG[-1] += ')'
                            isopen = False
                        arg = '(V*)'
                    else:
                        if ancestor != last_ancestor:
                            if isopen:
                                ARG[-1] += ')'                    
                            arg = '({:}*'.format(Y[ancestor]) if Y[ancestor] != '-' else '*'
                            isopen = True
                        else:
                            arg = '*'
                    ARG.append(arg)
                    last_ancestor = ancestor
                    _dtree_refresh(G)

                if isopen:
                    ARG[-2] += ')'


            prev_prop = proposition
            prev_time = t

        lb = ub
        ub = prev_time + 1  # ub must be inclusive
        last_ancestor = None
        G, root = _dtree_build(Dtree, ID, lb, ub)

        for i in range(lb, ub):
            subroot = root_d[prev_prop]
            q = deque(list())
            _, ancestor = _dtree_dfs(G, root, i, q)
            print('i:{:}\tID[i]:{:}\tancestor:{:}'.format(i, ID[i], ancestor))
            if ID[i] == subroot: # verb found
                if last_ancestor is not None:
                    ARG[-1] += ')'
                    isopen = False
                arg = '(V*)'
            else:
                if ancestor != last_ancestor:
                    arg = '({:}*'.format(Y[ancestor]) if Y[ancestor] != '-' else '*'
                    isopen = True
                else:
                    arg = '*'
            ARG.append(arg)
            last_ancestor = ancestor
            _dtree_refresh(G)
        if isopen:
            ARG[-2] += ')'
        return ARG


def _dtree_dfs(G, u, i, q):
    '''
        Returns true is u is ancestor of i
    '''
    G.nodes[u]['discovered'] = True
    q.append(u)

    # current node u is target node i
    if i == u:
        try:
            ancestor = q.popleft() # root
            ancestor = q.popleft() # first child
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


def _dtree_build(Dtree, ID, lb, ub):
    G = nx.Graph()
    root = None
    for i in range(lb, ub):
        G.add_node(i, discovered=False)

    for i in range(lb, ub):
        v = Dtree[i]
        u = ID[i]
        if v == 0:
            root = i
        else:
            G.add_edge(i, (v - u) + i)

    return G, root

#     def run(self):
#         '''
#             Computes the distance to the target predicate
#         '''
#         # defines output data structure
#         self.kernel = defaultdict(OrderedDict)

#         # finds predicate time 
#         predicate_d = _predicatedict(self.db)
#         lb = 0
#         ub = 0
#         prev_prop = -1
#         prev_time = -1
#         process = False
#         for time, proposition in self.db['P'].items():
#             if prev_prop < proposition:
#                 if prev_prop > 0:
#                     lb = ub
#                     ub = prev_time + 1  # ub must be inclusive
#                     process = True

#             if process:
#                 G, root = self._build(lb, ub)
#                 for i in range(lb, ub):
#                     # Find children, parent and grand-parent
#                     result = self._make_lookupnodes()
#                     q = deque(list())
#                     self._dfs_lookup(G, root, i, q, result)
#                     for key, nodeidx in result.items():
#                         for col in self.columns:
#                             new_key = '{:}_{:}'.format(col, key).upper()
#                             if nodeidx is None:
#                                 self.kernel[new_key][i] = None
#                             else:
#                                 self.kernel[new_key][i] = self.db[col][nodeidx]

#                     # Find path to predicate
#                     self._refresh(G)
#                     result = {}
#                     q = deque(list())
#                     pred = predicate_d[prev_prop]
#                     self._dfs_path(G, i, pred, q, result)

#                     for key, nodeidx in result.items():
#                         for col in self.columns:
#                             if col in ('GPOS', 'FUNC'):
#                                 _key = key.split('_')[0]
#                                 new_key = '{:}_{:}'.format(col, _key).upper()
#                                 if nodeidx is None:
#                                     self.kernel[new_key][i] = None
#                                 else:
#                                     self.kernel[new_key][i] = self.db[col][nodeidx]

#                     self._refresh(G)

#             process = False
#             prev_prop = proposition
#             prev_time = time

#         return self.kernel
    # def _make_lookupnodes(self):
    #     _list_keys = ['parent', 'grand_parent', 'child_1', 'child_2', 'child_3']
    #     return dict.fromkeys(_list_keys)

    # def _update_lookupnodes(self, children_l, ancestors_q, lookup_nodes):
    #     self._update_ancestors(ancestors_q, lookup_nodes)
    #     self._update_children(children_l, lookup_nodes)

    # def _update_path(self, ancestors_q, lookup_nodes):
    #     for i, nidx in enumerate(ancestors_q):
    #         _key = '{:02d}_node'.format(i)
    #         lookup_nodes[_key] = nidx

    # def _update_ancestors(self, ancestors_q, lookup_nodes):
    #     try:
    #         lookup_nodes['parent'] = ancestors_q.pop()
    #         lookup_nodes['grand_parent'] = ancestors_q.pop()
    #     except IndexError:
    #         pass

    # def _update_children(self, children_l, lookup_nodes):
    #     n = 0
    #     for v in children_l:
    #         if (not v == lookup_nodes['parent']):
    #             key = 'child_{:}'.format(n + 1)
    #             lookup_nodes[key] = v
    #             n += 1
    #         if n == 3:
    #             break

    # def _dfs_lookup(self, G, u, i, q, lookup_nodes):
    #     G.nodes[u]['discovered'] = True
    #     # updates ancestors if target i is undiscovered
    #     if not G.nodes[i]['discovered']:
    #         q.append(u)

    #     # current node u is target node i
    #     if i == u:
    #         self._update_lookupnodes(G.neighbors(u), q, lookup_nodes)
    #         return False
    #     else:
    #         # keep looking
    #         for v in G.neighbors(u):
    #             if not G.node[v]['discovered']:
    #                 search = self._dfs_lookup(G, v, i, q, lookup_nodes)
    #                 if not search:
    #                     return False

    #     if not G.nodes[i]['discovered']:
    #         q.pop()
    #     return True



    # def _crosssection(self, idx):
    #     list_keys = list(self.db.keys())
    #     d = {key: self.db[key][idx] for key in list_keys}
    #     d['discovered'] = False
    #     return d



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

    P = [1] * 33
    ID = list(range(1, 34))
    PRED = ['-'] * 33
    PRED[4] = 'revelar'
    Dtree = [5, 5, 2, 3, 0, 7, 5, 7, 7,
             25, 12, 10, 12, 25, 17, 17, 25,
             17, 20, 17, 17, 17, 24, 22, 7,
             27, 25, 25, 28, 31, 29, 31, 5]
    Y = ['-'] * 33
    Y[1] = 'A0'
    Y[6] = 'A1'

    P += [2] * 33
    ID += list(range(1, 34))
    PRED += ['recusar' if i == 9 else '-' for i in range(33)] 
    # PRED[32 + 9] = 'revelar'
    Dtree += [5, 5, 2, 3, 0, 7, 5, 7, 7,
             25, 12, 10, 12, 25, 17, 17, 25,
             17, 20, 17, 17, 17, 24, 22, 7,
             27, 25, 25, 28, 31, 29, 31, 5]
    # Y += ['-'] * 32
    # Y[32 + 12] = 'A1'
    Y += ['A1' if i == 11 else '-' for i in range(33)] 
    ARG = propbankbr_y2arg2(P, ID, PRED, Dtree, Y)
    import code; code.interact(local=dict(globals(), **locals()))

    # arg2t = propbankbr_arg2t(propositions, arguments)
    # t2arg = propbankbr_t2arg(propositions, arg2t)
    # arg2r = propbankbr_arg2r(arguments)
    # r2arg = propbankbr_r2arg(forms, propositions, arg2r)
    # with open('test_arguments.csv', mode='w') as f:
    #     f.write(',ARG,T,T2ARG,R,R2ARG\n')
    #     for i, arg in enumerate(arguments):
    #         line = '{:},{:},{:},{:},{:},{:}\n'.format(i, arg, arg2t[i], t2arg[i], arg2r[i], r2arg[i])
    #         f.write(line)
