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

from collections import OrderedDict
# from config import DATASET_TRAIN_SIZE, DATASET_VALID_SIZE, DATASET_TEST_SIZE
import models.utils
import datasets as br

PEARL_SRLEVAL_PATH='./srlconll-1.1/bin/srl-eval.pl'


class EvaluatorConll(object):

    def __init__(self, db, idx2lex, target_dir='./'):
        '''
            args:
            returns:
        '''
        self.db = db
        self.idx2lex = idx2lex
        self.target_dir = target_dir
        self.target_columns = ('ARG', 'HEAD', 'T', 'IOB')
        self.props_dict = {}
        self._refresh()


    def evaluate_tensor(self, prefix, index_tensor, predictions_tensor, len_tensor, target_column, hparams):
        '''
            Evaluates the conll scripts returning total precision, recall and F1
                if self.target_dir is set will also save conll-<prefix>.txt@self.target_dir/<hparams>/

            args:
                index_tensor        .:
                predictions_tensor  .:
                len_tensor          .:

            returns:
                prec                .: float<> precision
                rec                 .: float<> recall 
                f1                  .: float<> F1 score
        '''
        if target_column not in self.target_columns:
           raise ValueError('target_column must be in {:} got target_column {:}'.format(self.target_columns, target_column))

        self.props_dict = self._tensor2dict(index_tensor, predictions_tensor, len_tensor, target_column)

        if target_column in ('IOB',):
            self._iob2arg()
        elif target_column in ('HEAD',):
            self._head2arg()
        elif target_column in ('T',):
            self._t2arg()

        self.evaluate(prefix, self.props_dict, hparams)

    def evaluate(self, filename, props, hparams):
        '''
            Evaluates the conll scripts returning total precision, recall and F1
                if self.target_dir is set will also save conll-<prefix>.txt@self.target_dir/<hparams>/

            args:
                prefix            .: list<string>
                props             .: list<string>
                hparams           .: list<string> 

            returns:
                prec            .: float<> precision
                rec       .: float<> recall 
                f1        .: float<> F1 score
        '''
        #Resets state
        self._refresh()

        #Generate GOLD standard labels
        indexes = sorted(list(props.keys()))        
        gold_props = {i: self.idx2lex['ARG'][self.db['ARG'][i]] for i in indexes}

        #Stores GOLD standard labels
        gold_path = self._store(filename, 'gold', self.db, self.idx2lex, gold_props, hparams)

        #Stores evaluation labels
        eval_path = self._store(filename, 'eval', self.db, self.idx2lex, props, hparams)

        #Runs official conll 2005 shared task script
        pipe = subprocess.Popen(['perl',PEARL_SRLEVAL_PATH, gold_path, eval_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out is a byte with each line separated by \n
        #ers is stderr      
        txt, err = pipe.communicate()
        self.txt = txt.decode('UTF-8')
        self.err = err.decode('UTF-8')

        if (self.err):
            print('srl-eval.pl says:\t{:}'.format(self.err))

        #Parses final txt
        self._parse(self.txt)

        #Stores target_dir/<hparams>/prefix
        target_dir = self._get_dir(hparams)
        target_path = '{:}/{:}.conll'.format(target_dir, filename)
        with open(target_path, 'w+') as f:
            f.write(self.txt)

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
        self.txt = ''
        self.err = ''
        self.f1 = -1
        self.precision = -1
        self.recall = -1

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

    def _store(self, ds_type, prediction_type, db, lexicons, props, hparams):
        '''
            Stores props and stats into target_dir 
        '''
        # hparam_string = self._make_hparam_string(**hparams)
        # target_dir += '/{:}/'.format(hparam_string)
        # if not os.path.isdir(target_dir):
        #     os.mkdir(target_dir)
        target_dir = self._get_dir(hparams)
        target_path = '{:}/{:}-{:}.props'.format(target_dir, ds_type, prediction_type)

        p = db['P'][min(props)]
        with open(target_path, mode='w+') as f:
            for idx, prop in props.items():
                if db['P'][idx] != p:
                    f.write('\n')
                    p = db['P'][idx]
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

    def _get_dir(self, hparams):
        hparam_string = self._make_hparam_string(**hparams)
        target_dir = '{:}{:}/'.format(self.target_dir, hparam_string)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        return target_dir

    def _get_goldindex_list(self, ds_type):
        # if ds_type in ['train']:
        #     gold_index_list = [s for s in range(0, DATASET_TRAIN_SIZE)]
        # elif ds_type in ['valid']:
        #     gold_index_list = [s for s in range(DATASET_TRAIN_SIZE,
        #                                         DATASET_TRAIN_SIZE + DATASET_VALID_SIZE)]
        # elif ds_type in ['test']:
        #     gold_index_list = [s for s in range(DATASET_TRAIN_SIZE + DATASET_VALID_SIZE,
        #                                         DATASET_TRAIN_SIZE + DATASET_VALID_SIZE + DATASET_TEST_SIZE)]
        # else:
        #     raise ValueError('{:} unknown dataset type'.format(ds_type))
        lb, ub = models.utils.get_db_bounds(ds_type)
        return [i for i in range(lb, ub + 1)]

    def _tensor2dict(self, index_tensor, predictions_tensor, len_tensor, target_column):
        index = [item
                 for i, sublist in enumerate(index_tensor.tolist())
                 for j, item in enumerate(sublist) if j < len_tensor[i]]

        values = [self.idx2lex[target_column][int(item)]
                  for i, sublist in enumerate(predictions_tensor.tolist())
                  for j, item in enumerate(sublist) if j < len_tensor[i]]

        _zip_list = sorted(zip(index, values), key=lambda x: x[0])
        self.props_dict = OrderedDict(_zip_list)

        return self.props_dict

    def _t2arg(self):
        propositions = {idx: self.db['P'][idx] for idx in self.props_dict}


        ARG = br.propbankbr_t2arg(propositions.values(), self.props_dict.values())
        self.props_dict = OrderedDict(sorted(zip(self.props_dict.keys(), ARG), key=lambda x: x[0]))

        return self.props_dict

    def _iob2arg(self):
        propositions = {idx: self.db['P'][idx] for idx in self.props_dict}


        ARG = br.propbankbr_iob2arg(propositions.values(), self.props_dict.values())
        self.props_dict = OrderedDict(sorted(zip(self.props_dict.keys(), ARG), key=lambda x: x[0]))

        return self.props_dict

    def _head2arg(self):
        propositions = {idx: self.db['P'][idx] for idx in self.props_dict}
        head = ['*' if head_ == '-' else '({:}*)'.format(head_)
                for _, head_ in self.props_dict.items()]

        self.props_dict = OrderedDict(sorted(zip(self.props_dict.keys(), head), key=lambda x: x[0]))

        return self.props_dict