'''
    This module works a wrapper for pearl's conll task evaluation script

    Created on Mar 15, 2018
    
    @author: Varela


    ref:
    
    CONLL 2005 SHARED TASK
        HOME: http://www.lsi.upc.edu/~srlconll/
        SOFTWARE: http://www.lsi.upc.edu/~srlconll/soft.html

'''
import sys
sys.path.append('../datasets/')
import os
import subprocess
import string
import pandas as pd
import pickle
import re

from config import DATASET_TRAIN_SIZE, DATASET_VALID_SIZE, DATASET_TEST_SIZE


PEARL_SRLEVAL_PATH='./srlconll-1.1/bin/srl-eval.pl'

class EvaluatorConll2(object):

    def __init__(self, db, idx2lex, target_dir='./'):
        '''
            args:
            returns:
        '''             
        self.db = db
        self.idx2lex = idx2lex
        self.target_dir = target_dir

        self._refresh()

    def evaluate(self, ds_type, props, hparams):
        '''
            Evaluates the conll scripts returning total precision, recall and F1
                if self.target_dir is set will also save conll.txt@self.target_dir

            Performs a 6-step procedure in order to use the script evaluation
            1) Formats      .: inputs in order to obtain proper conll format ()
            2) Saves      .:  two tmp files tmpgold.txt and tmpy.txt on self.root_dir.
            3) Run            .:  the perl script using subprocess module.
            4) Parses     .:  parses results from 3 in variables self.f1, self.prec, self.rec. 
            5) Stores     .:  stores results from 3 in self.target_dir 
            6) Cleans     .:  files left from step 2.
                
            args:
                PRED            .: list<string> predicates according to PRED column
                T               .: list<string> target according to ARG column
                Y               .: list<string> 
            returns:
                prec            .: float<> precision
                rec       .: float<> recall 
                f1        .: float<> F1 score
        '''

        #Resets state
        self._refresh()
        #Step 1 - Transforms columns into with args and predictions into a dictionary
        # ready with conll format
        if ds_type in ['train']:
            gold_index_list = [s for s in range(0, DATASET_TRAIN_SIZE)]
        elif ds_type in ['valid']:
            gold_index_list = [s for s in range(DATASET_TRAIN_SIZE,
                                                DATASET_TRAIN_SIZE + DATASET_VALID_SIZE)]
        elif ds_type in ['test']:
            gold_index_list = [s for s in range(DATASET_TRAIN_SIZE + DATASET_VALID_SIZE,
                                                DATASET_TRAIN_SIZE + DATASET_VALID_SIZE + DATASET_TEST_SIZE)]
        else:
            raise ValueError('{:} unknown dataset type'.format(ds_type))
        # df_eval, df_gold = self._conll_format(Y)        

        #Step 2 - Uses target dir to save files     
        # eval_path, gold_path= self._store(df_eval, df_gold)
        gold_props = {i: self.idx2lex['ARG'][self.db['ARG'][i]] 
                      for i, s in self.db['S'].items() if s in set(gold_index_list)}

        # import code; code.interact(local=dict(globals(), **locals()))
        gold_path = self._store(ds_type, 'gold',self.db, self.idx2lex, gold_props, hparams, self.target_dir)
        
        
        # eval_props = {i: self.idx2lex[pred] for i, pred in props_dict.items()}
        eval_path = self._store(ds_type, 'eval', self.db, self.idx2lex, props, hparams, self.target_dir)

        #Step 3 - Popen runs the pearl script storing in the variable PIPE
        pipe= subprocess.Popen(['perl',PEARL_SRLEVAL_PATH, gold_path, eval_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out is a byte with each line separated by \n
        #ers is stderr      
        txt, err = pipe.communicate()
        self.txt = txt.decode('UTF-8')
        self.err = err.decode('UTF-8')

        if (self.err):
            print('srl-eval.pl says:\t{:}'.format(self.err))

        #Step 4 - Parse     
        self._parse(self.txt)
        if store:
            # Step 5 - Stores
            target_path= '{:}conllscore_{:}.txt'.format(self.target_dir, self.ds_type)
            with open(target_path, 'w+') as f:
                f.write(self.txt)
        else:
            # Step 6 - Removes tmp
            try:
                os.remove(eval_path)
            except OSError:
                pass

            try:
                os.remove(gold_path)
            except OSError:
                pass

    def evaluate_fromconllfile(self, target_path):
        '''
            Opens up a conll file and parses it

            args:
                target_path .: string filename + dir 
        '''
        self._refresh()

        with open(target_path, 'r') as f: 
            self.txt = f.read()
        f.close()

        self._parse(self.txt)

    def _refresh(self):
        self.num_propositions = -1
        self.num_sentences = -1
        self.perc_propositions = -1
        self.txt = ''
        self.err = ''
        self.f1 = -1
        self.precision = -1
        self.recall = -1

    # def _conll_format(self, Y):
    #     df_eval = conll_with_dicts(self.S, self.P, self.PRED, Y, True)
    #     df_gold = conll_with_dicts(self.S, self.P, self.PRED, self.ARG, True)
    #     return df_eval, df_gold

    # def _store(self, df_eval, df_gold):
    #     eval_path = self.target_dir + '{:}eval.txt'.format(self.ds_type)
    #     df_eval.to_csv(eval_path, sep='\t', index=False, header=False)


    #     gold_path = self.target_dir + '{:}gold.txt'.format(self.ds_type)
    #     df_gold.to_csv(gold_path, sep ='\t', index=False, header=False)
    #     return eval_path, gold_path

    def _parse(self, txt):
        '''
        Parses srlconll-1.1/bin/srl-eval.pl script output text 



        args:
            txt .: string with lines separated by \n and fields separated by tabs

        returns:

        example:
        Number of Sentences    :         326
        Number of Propositions :         553
        Percentage of perfect props :   4.70
                          corr.  excess  missed    prec.    rec.      F1
        ------------------------------------------------------------
                 Overall      398    2068     866    16.14   31.49   21.34
        ----------
                    A0      124     285     130    30.32   48.82   37.41
                    A1      202    1312     288    13.34   41.22   20.16
                    A2       18     179     169     9.14    9.63    9.37
                    A3        1      14      15     6.67    6.25    6.45
                    A4        2      14       9    12.50   18.18   14.81
                AM-ADV        4      19      19    17.39   17.39   17.39
                AM-CAU        0      16      17     0.00    0.00    0.00
                AM-DIR        0       0       1     0.00    0.00    0.00
                AM-DIS        6      17      20    26.09   23.08   24.49
                AM-EXT        0       3       5     0.00    0.00    0.00
                AM-LOC        9      65      46    12.16   16.36   13.95
                AM-MNR        0      24      28     0.00    0.00    0.00
                AM-NEG       10       5      24    66.67   29.41   40.82
                AM-PNC        0       9       9     0.00    0.00    0.00
                AM-PRD        0      15      18     0.00    0.00    0.00
                AM-TMP       22      91      68    19.47   24.44   21.67
        ------------------------------------------------------------
                     V      457      32      96    93.46   82.64   87.72
        ------------------------------------------------------------

        '''
        lines = txt.split('\n')
        for i, line in enumerate(lines):
            if (i == 0):
                self.num_sentences = int(line.split(':')[-1])
            if (i == 1):
                self.num_propositions = int(line.split(':')[-1])
            if (i == 2):
                self.perc_propositions = float(line.split(':')[-1])
            if (i == 6):
                self.precision, self.recall, self.f1 = map(float, line.split()[-3:])
                break

    def _store(self, ds_type, prediction_type, db, lexicons, props, hparams, target_dir):
        '''
            Stores props and stats into target_dir 
        '''
        hparam_string = self._make_hparam_string(**hparams)
        target_dir += '/{:}/'.format(hparam_string)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        target_path = '{:}/{:}-{:}.props'.format(target_dir, ds_type, prediction_type)

        p = db['P'][min(props)]
        with open(target_path, mode='w+') as f:
            for idx, prop in props.items():
                if db['P'][idx] != p:
                    f.write('\n')
                    p = db['P'][idx]
                # import code; code.interact(local=dict(globals(), **locals()))
                f.write('{:}\t{:}\n'.format(lexicons['PRED'][db['PRED'][idx]], prop))
        f.close()
        return target_path

    def _make_hparam_string(self, learning_rate=1 * 1e-3, hidden_size=[32, 32], ctx_p=1, embeddings_id='', **kwargs):
        '''
            Makes a directory name from hyper params
            args:
            returns:

        '''        
        hs = re.sub(r', ','x', re.sub(r'\[|\]', '', str(hidden_size)))
        hparam_string= 'lr{:.2e}_hs{:}_ctx-p{:d}'.format(float(learning_rate), hs, int(ctx_p))
        if embeddings_id:
            hparam_string += '_{:}'.format(embeddings_id)
        return hparam_string


class EvaluatorConll(object):

    def __init__(self, ds_type, S, P, PRED, ARG, target_dir='./'):
        '''
            args:
                ds_type
            S           .:  dict<int,int> keys are the index, values are sentences
            P       .:  dict<int,int> keys are the index, values are propositions
            PRED    .:  dict<int,str> keys are the index, values are verbs/ predicates
            ARG     .:  dict<int,str> keys are the index, values are ARG
        '''             
        self.ds_type = ds_type
        self.S = S 
        self.P = P 
        self.PRED = PRED
        self.ARG = ARG
        self.target_dir = target_dir

        self._refresh()

    def evaluate(self, Y, store=False):
        '''
            Evaluates the conll scripts returning total precision, recall and F1
                if self.target_dir is set will also save conll.txt@self.target_dir
            Performs a 6-step procedure in order to use the script evaluation
            1) Formats      .: inputs in order to obtain proper conll format ()
            2) Saves      .:  two tmp files tmpgold.txt and tmpy.txt on self.root_dir.
            3) Run            .:  the perl script using subprocess module.
            4) Parses     .:  parses results from 3 in variables self.f1, self.prec, self.rec. 
            5) Stores     .:  stores results from 3 in self.target_dir 
            6) Cleans     .:  files left from step 2.
                
            args:
                PRED            .: list<string> predicates according to PRED column
                T               .: list<string> target according to ARG column
                Y               .: list<string> 
            returns:
                prec            .: float<> precision
                rec       .: float<> recall 
                f1        .: float<> F1 score
        '''

        #Resets state
        self._refresh()
        #Step 1 - Transforms columns into with args and predictions into a dictionary
        # ready with conll format
        df_eval, df_gold = self._conll_format(Y)        

        #Step 2 - Uses target dir to save files     
        eval_path, gold_path= self._store(df_eval, df_gold)
        
        #Step 3 - Popen runs the pearl script storing in the variable PIPE
        pipe= subprocess.Popen(['perl',PEARL_SRLEVAL_PATH, gold_path, eval_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out is a byte with each line separated by \n
        #ers is stderr      
        txt, err = pipe.communicate()
        self.txt= txt.decode('UTF-8')
        self.err= err.decode('UTF-8')

        if (self.err):
            print('srl-eval.pl says:\t{:}'.format(self.err))

        #Step 4 - Parse     
        self._parse(self.txt)
        if store:
            # Step 5 - Stores
            target_path= '{:}conllscore_{:}.txt'.format(self.target_dir, self.ds_type)
            with open(target_path, 'w+') as f:
                f.write(self.txt)
        else:
            # Step 6 - Removes tmp
            try:
                os.remove(eval_path)
            except OSError:
                pass

            try:
                os.remove(gold_path)
            except OSError:
                pass

    def evaluate_fromconllfile(self, target_path):
        '''
            Opens up a conll file and parses it
            args:
                target_path .: string filename + dir 
        '''
        self._refresh()

        with open(target_path, 'r') as f: 
            self.txt = f.read()
        f.close()

        self._parse(self.txt)

    def evaluate_fromliblinear(self, target_path, mapper, target_column='T'):
        '''
            Opens up a conll file and parses it
            args:
                target_path .: string filename + dir 
        '''
        self._refresh()

        with open(target_path, 'rb') as f:
            d = pickle.load(f)
        f.close()


        if len(d) != len(self.ARG):
            raise ValueError('number of predictions must match targets')
        else:
            self.target_dir = '/'.join(target_path.split('/')[:-1])
            self.target_dir += '/'
            if target_column in ('T'):
                Y = mapper.define(d, 'IDX').map()
            else:
                Y = mapper.define(d, target_column).map()

            self.evaluate(Y, store=True)

    def _refresh(self):
        self.num_propositions = -1
        self.num_sentences = -1
        self.perc_propositions = -1
        self.txt = ''
        self.err = ''
        self.f1 = -1
        self.precision = -1
        self.recall = -1

    def _conll_format(self, Y):
        df_eval = conll_with_dicts(self.S, self.P, self.PRED, Y, True)
        df_gold = conll_with_dicts(self.S, self.P, self.PRED, self.ARG, True)
        return df_eval, df_gold

    def _store(self, df_eval, df_gold):
        eval_path = self.target_dir + '{:}eval.txt'.format(self.ds_type)
        df_eval.to_csv(eval_path, sep='\t', index=False, header=False)


        gold_path = self.target_dir + '{:}gold.txt'.format(self.ds_type)
        df_gold.to_csv(gold_path, sep ='\t', index=False, header=False)
        return eval_path, gold_path

    def _parse(self, txt):
        '''
        Parses srlconll-1.1/bin/srl-eval.pl script output text 
        args:
            txt .: string with lines separated by \n and fields separated by tabs
        returns:
        example:
        Number of Sentences    :         326
        Number of Propositions :         553
        Percentage of perfect props :   4.70
                          corr.  excess  missed    prec.    rec.      F1
        ------------------------------------------------------------
                 Overall      398    2068     866    16.14   31.49   21.34
        ----------
                    A0      124     285     130    30.32   48.82   37.41
                    A1      202    1312     288    13.34   41.22   20.16
                    A2       18     179     169     9.14    9.63    9.37
                    A3        1      14      15     6.67    6.25    6.45
                    A4        2      14       9    12.50   18.18   14.81
                AM-ADV        4      19      19    17.39   17.39   17.39
                AM-CAU        0      16      17     0.00    0.00    0.00
                AM-DIR        0       0       1     0.00    0.00    0.00
                AM-DIS        6      17      20    26.09   23.08   24.49
                AM-EXT        0       3       5     0.00    0.00    0.00
                AM-LOC        9      65      46    12.16   16.36   13.95
                AM-MNR        0      24      28     0.00    0.00    0.00
                AM-NEG       10       5      24    66.67   29.41   40.82
                AM-PNC        0       9       9     0.00    0.00    0.00
                AM-PRD        0      15      18     0.00    0.00    0.00
                AM-TMP       22      91      68    19.47   24.44   21.67
        ------------------------------------------------------------
                     V      457      32      96    93.46   82.64   87.72
        ------------------------------------------------------------
        '''
        lines = txt.split('\n')
        for i, line in enumerate(lines):
            if (i == 0):
                self.num_sentences = int(line.split(':')[-1])
            if (i == 1):
                self.num_propositions = int(line.split(':')[-1])
            if (i == 2):
                self.perc_propositions = float(line.split(':')[-1])
            if (i == 6):
                self.precision, self.recall, self.f1 = map(float, line.split()[-3:])
                break


def conll_with_dicts(S, P, PRED, Y, to_frame=True):
  '''
    Converts a dataset to conll format - promotes kind of an horizontal stack,
        in which we add each proposition within the sentence, as a new column or key
    
    args:
        S           .:  dict<int,int> keys are the index, values are sentences
        P       .:  dict<int,int> keys are the index, values are propositions
        PRED    .:  dict<int,str> keys are the index, values are verbs/ predicates
        Y       .:  dict<int,str> keys are the index, values are Y is 'ARG', 'T', 'ARG'
            to_frame .: bool if true converts to the output to dataframe
        returns:
            d_conll .: dict<str,dict<int,str>> outer keys are columns 'PRED', 'ARG0', 'ARG1', ..., 'ARGN'
                        inner keys are new indexes ( over the sentences)
                        values are 'PRED', 'ARG0', 'ARG1', ..., 'ARGN'
                        or dataframe where columns are db columns and rows are tokens with sequence
        refs: 
            http://localhost:8888/notebooks/05-evaluations_conll.ipynb              
  '''
  d_conll = {}
  index1 = 0
  index0 = 0 # marks the beginning of a new SENTENCE
  prev_p = -1
  prev_s = -1
  pps = -1 #propostion per sentence  
  first = True
  for index, s in S.items():
    p = P[index]
    pred = PRED[index]
    y = Y[index]
    if p != prev_p and s != prev_s: #New Sentence and new proposition
        pps = 0  # fills ARG0  
        # conll format .: skip a row for each new sentence after the first
        if not(first):
            for colname in d_conll:
                d_conll[colname][index1]=''
            index1 += 1
        index0 = index1 #Stores the beginning of the sentence                
    elif p != prev_p:#New proposition
        pps+=1  #  updates column to write
        index1=index0 # back to the first key
        
    argkey = 'ARG{:}'.format(pps)    
    if not(argkey in d_conll):
        if first:        
            d_conll['PRED']={}
            first=False
        d_conll[argkey]={}


    #updates predicate if index1 is unseen 
    if not(index1 in d_conll['PRED']) or not(pred =='-'):
        d_conll['PRED'][index1]=pred
        
    d_conll[argkey][index1]=y #            
    prev_p=p
    prev_s=s    
    index1+=1
  
  result= d_conll
  if (to_frame):
    result = pd.DataFrame.from_dict(d_conll , orient='columns') 
    l= len(d_conll)
    usecols= ['PRED']  + ['ARG{:}'.format(i) for i in range(l-1)]# reorder columns to match format
    result= result[usecols]
  return result 
