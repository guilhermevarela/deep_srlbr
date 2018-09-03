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
import os
from subprocess import PIPE, Popen
import string
import pandas as pd
import pickle
import re
import tempfile

from collections import OrderedDict

sys.path.append('../datasets/')
sys.path.append('..')
import utils
import datasets as br


PEARL_SRL04_PATH = 'srlconll04/srl-eval.pl'
PEARL_SRL05_PATH = 'srlconll05/bin/srl-eval.pl'


def to_conll(atuple):
    txt = '{:}'.format(atuple[0])
    txt += '\t{:}' * (len(atuple) - 1)
    txt += '\n'

    return txt.format(*atuple[1:])


def evaluate(gold_list, eval_list,
             verbose=True, script_version='05', file_dir=None,
             file_name='tmp'):
    '''[summary]

    [description]

    Arguments:
        gold_list {[type]} -- [description]
        eval_list {[type]} -- [description]

    Keyword Arguments:
        verbose {bool} -- [description] (default: {True})
        script_version {str} -- [description] (default: {'05'})
        file_dir {[type]} -- [description] (default: {None})
        name {str} -- [description] (default: {'tmp'})

    Returns:
        [type] -- [description]
    '''
    # Solves directory
    if file_dir is None:
        file_dir = tempfile.gettempdir() + '/'

    # Solves script version
    if script_version not in ('04', '05',):
        msg = 'script version {:}'.format(script_version)
        msg += 'invalid only `04` and `05` allowed'
        raise ValueError(msg)

    else:
        if script_version == '04':
            script_path = PEARL_SRL04_PATH
        else:
            script_path = PEARL_SRL05_PATH

    gold_path = '{:}{:}-gold.props'.format(file_dir, file_name)
    eval_path = '{:}{:}-eval.props'.format(file_dir, file_name)
    conll_path = '{:}{:}.conll'.format(file_dir, file_name)

    with open(gold_path, mode='w') as f:
        for gold_ in gold_list:
            if gold_ is None:
                f.write('\n')
            else:
                f.write(to_conll(gold_))

    with open(eval_path, mode='w') as f:
        for gold_ in eval_list:
            if gold_ is None:
                f.write('\n')
            else:
                f.write(to_conll(gold_))

    cmd_list = ['perl', script_path, gold_path, eval_path]
    pipe = Popen(cmd_list, stdout=PIPE, stderr=PIPE)

    txt, err = pipe.communicate()
    txt = txt.decode('UTF-8')
    err = err.decode('UTF-8')

    if verbose:
        print(txt)
        print(conll_path)
        with open(conll_path, mode='w') as f:
            f.write(txt)

    # overall is a summary from the list
    # is the seventh line
    lines_list = txt.split('\n')

    # get the numbers from the row
    overall_list = re.findall(r'[-+]?[0-9]*\.?[0-9]+.', lines_list[6])
    f1 = float(overall_list[-1])

    return f1


def _props_file2zip_list(file_path):
    '''Opens a props file and returns as a zipped list

    Arguments:
        file_path {str} -- String representing a props file full path

    '''
    zip_list = []

    def trim(txt):
        return str(txt).rstrip().lstrip()

    def invalid(txt):
        return re.sub('\n', '', txt)

    with open(file_path, mode='r') as f:
        for line in f.readlines():
            if len(line) > 1:
                line_list = [trim(val)
                             for val in invalid(line).split('\t')]
                zip_list.append(tuple(line_list))
            else:
                zip_list.append(None)
    return zip_list


def _filter_list(zip_list, keep_list):
    '''Keeps only tags in keep list

    Erase all tags not in keep_list 
    * A0, A1, AM-NEG, V

    Arguments:
        zip_list {list{tuple}} -- zip_list is a list that each item is 
            a list.

    '''
    filter_list = []
    first = True
    for tuple_ in zip_list:
        if tuple_ is not None:
            if first:
                open_labels_ = [False] * (len(tuple_) - 1)
                first = False
            list_ = list(tuple_)
            for column_, value_ in enumerate(list_[1:]):
                if any([val_ in value_ for val_ in keep_list]):
                    list_[column_ + 1] = value_

                    open_labels_[column_] = ')' not in value_
                elif value_ == '*)' and open_labels_[column_]:
                    list_[column_ + 1] = value_
                    open_labels_[column_] = False
                else:
                    list_[column_ + 1] = '*'
            tuple_ = tuple(list_)
        else:
            first = True

        filter_list.append(tuple_)

    return filter_list


class EvaluatorConll(object):

    def __init__(self, db, idx2lex, target_dir=None):
        self.db = db
        self.idx2lex = idx2lex
        self.target_dir = target_dir
        self.target_columns = ('ARG', 'HEAD', 'T', 'IOB')
        self.props_dict = {}
        self._refresh()

    @staticmethod
    def evaluate_frompropositions(gold_path, predicted_path,
                                  file_name, verbose=True, keep_list=None,
                                  script_version='05'):
        '''Wraps pearl calls to CoNLL Shared Task 2005 script

            Evaluates the conll scripts returning total precision, recall and F1
            *accepts a filter list for restricting arguments
            *saves a temporary file and deletes it
        Arguments:
            gold_path {[type]} -- [description]
            predicted_path {[type]} -- [description]
            file_name {[type]} -- [description]
        
        Keyword Arguments:
            verbose {bool} -- [description] (default: {True})
            keep_list {[type]} -- [description] (default: {None})
            script_version {str} -- [description] (default: {'05'})

        Returns:
            [type] -- [description]
        '''
        gold_list = _props_file2zip_list(gold_path)
        eval_list = _props_file2zip_list(predicted_path)

        # filter
        if keep_list is not None:
            gold_list = _filter_list(gold_list, keep_list)
            eval_list = _filter_list(eval_list, keep_list)

        # converts CoNLL 2005 ST to CoNLL 2004 ST
        if script_version == '04':
            gold_list = br.propbankbr_arg2se(gold_list)
            eval_list = br.propbankbr_arg2se(eval_list)

        f1 = evaluate(gold_list, eval_list,
                      verbose=True, script_version=script_version, file_dir=None,
                      file_name=file_name)

        return f1

    def evaluate_tensor(self, prefix,
                        index_tensor, predictions_tensor, len_tensor,
                        target_column, hparams, script_version='04'):
        ''''Wraps pearl calls to CoNLL Shared Task 2004/2005 script

        Evaluates the conll scripts returning total precision, recall and F1
        if self.target_dir is set will also save conll-<prefix>.txt@self.target_dir/<hparams>/

        Arguments:
            prefix {[type]} -- [description]
            index_tensor {[type]} -- [description]
            predictions_tensor {[type]} -- [description]
            len_tensor {[type]} -- [description]
            target_column {[type]} -- [description]
            hparams {[type]} -- [description]

        Keyword Arguments:
            script_version {str} --  `04` for CoNLL 2004 (default: '04')
        '''
        if script_version not in ('04', '05'):
            msg_ = 'script_version: {:} must be in (`04`,`05`)'
            raise ValueError(msg_.format(script_version))

        if target_column not in self.target_columns:
            msg_ = 'target_column must be in {:} got target_column {:}'
            msg_ = msg_.format(self.target_columns, target_column)
            raise ValueError(msg_)

        self.props_dict = self._tensor2dict(index_tensor, predictions_tensor, len_tensor, target_column)

        if target_column in ('IOB',):
            self._iob2arg()
        elif target_column in ('HEAD',):
            self._head2arg()
        elif target_column in ('T',):
            self._t2arg()

        self.evaluate(prefix, self.props_dict,
                      hparams, script_version=script_version)

    def evaluate(self, filename, props, hparams, script_version='04'):
        '''Wraps pearl calls to CoNLL Shared Task 2004/2005 script

        Evaluates the conll scripts returning total precision, recall and F1
        if self.target_dir is set will also save conll-<prefix>.txt@self.target_dir/<hparams>/

        Arguments:
            filename {str} -- [description]
            props {dict} -- List keys are token index and values are the arguments.
            hparams {list} -- list of hyper parameters
        '''
        if script_version not in ('04', '05'):
            msg_ = 'script_version: {:} must be in (`04`,`05`)'
            raise ValueError(msg_.format(script_version))

        # Resets state
        self._refresh()

        # Generate GOLD standard labels
        indexes = sorted(list(props.keys()))

        gold_props = {i: self.idx2lex['ARG'][self.db['ARG'][i]]
                      for i in indexes}

        if script_version == '04':            
            gold_props = self._arg2start_end(gold_props)

        # Stores GOLD standard labels
        arg_list = [filename, 'gold', self.db,
                    self.idx2lex, gold_props, hparams]
        gold_path = self._store(*arg_list)

        if script_version == '04':
            eval_props = self._arg2start_end(props)
        else:
            eval_props = props

        # Stores evaluation labels
        arg_list = [filename, 'eval', self.db,
                    self.idx2lex, eval_props, hparams]

        eval_path = self._store(*arg_list)

        # Runs official conll 2005 shared task script
        if script_version  == '05':
            perl_path = PEARL_SRL05_PATH
        elif script_version == '04':
            perl_path = PEARL_SRL04_PATH

        perl_cmd = ['perl', perl_path, gold_path, eval_path]

        # Resets state
        pipe = Popen(perl_cmd, stdout=PIPE, stderr=PIPE)

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
        if self.target_dir is None:
            target_dir = self._get_dir(hparams)
        else:
            target_dir = self.target_dir
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
                p, r, f = map(float, line.split()[-3:])
                self.precision, self.recall, self.f1 = p, r, f
                break

    def _store(self, ds_type, prediction_type, db, lexicons, props, hparams):
        '''
            Stores props and stats into target_dir
        '''
        if self.target_dir is None:
            target_dir = self._get_dir(hparams)
        else:
            target_dir = self.target_dir

        target_path = '{:}/{:}-{:}.props'.format(
            target_dir,
            ds_type,
            prediction_type
        )

        p = db['P'][min(props)]
        with open(target_path, mode='w+') as f:
            for idx, prop in props.items():
                if db['P'][idx] != p:
                    f.write('\n')
                    p = db['P'][idx]
                f.write('{:}\t{:}\n'.format(lexicons['PRED'][db['PRED'][idx]], prop))

        f.close()
        return target_path

    def _make_hparam_string(self, learning_rate=1 * 1e-3,
                            hidden_size=[32, 32], ctx_p=1, embeddings_id='',
                            **kwargs):
        '''
            Makes a directory name from hyper params
            args:
            returns:

        '''
        # hidden size hparameter
        hs = re.sub(r'\[|\]', '', str(hidden_size))
        hs = re.sub(r', ', 'x', hs)

        hparam_args = (float(learning_rate), hs, int(ctx_p))
        hparam_string = 'lr{:.2e}_hs{:}_ctx-p{:d}'
        hparam_string = hparam_string.format(*hparam_args)
        if embeddings_id:
            hparam_string += '_{:}'.format(embeddings_id)
        return hparam_string

    def _get_dir(self, hparams):
        hparam_string = self._make_hparam_string(**hparams)
        target_dir = '{:}{:}/'.format('./', hparam_string)
        # Octal for read / write permissions
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir, 0o777)
        return target_dir

    def _get_goldindex_list(self, ds_type, version='1.0'):
        lb, ub = utils.get_db_bounds(ds_type, version='1.0')

        return [i for i in range(lb, ub)]

    def _tensor2dict(self, index_tensor,
                     predictions_tensor, len_tensor, target_column):

        index = [item
                 for i, sublist in enumerate(index_tensor.tolist())
                 for j, item in enumerate(sublist) if j < len_tensor[i]]

        values = [self.idx2lex[target_column][int(item)]
                  for i, sublist in enumerate(predictions_tensor.tolist())
                  for j, item in enumerate(sublist) if j < len_tensor[i]]

        zip_list = sorted(zip(index, values), key=lambda x: x[0])
        self.props_dict = OrderedDict(zip_list)

        return self.props_dict

    def _t2arg(self):
        propositions = {idx: self.db['P'][idx] for idx in self.props_dict}

        prop_list = propositions.values()
        arg_list = self.props_dict.values()

        ARG = br.propbankbr_t2arg(prop_list, arg_list)

        zip_list = zip(self.props_dict.keys(), ARG)
        self.props_dict = OrderedDict(sorted(zip_list, key=lambda x: x[0]))

        return self.props_dict

    def _iob2arg(self):
        propositions = {idx: self.db['P'][idx] for idx in self.props_dict}

        prop_list = propositions.values()
        arg_list = self.props_dict.values()

        ARG = br.propbankbr_iob2arg(prop_list, arg_list)
        zip_list = zip(self.props_dict.keys(), ARG)
        self.props_dict = OrderedDict(sorted(zip_list, key=lambda x: x[0]))

        return self.props_dict

    def _head2arg(self):
        head_list = ['*' if head_ == '-' else '({:}*)'.format(head_)
                     for _, head_ in self.props_dict.items()]

        zip_list = zip(self.props_dict.keys(), head_list)
        self.props_dict = OrderedDict(sorted(zip_list, key=lambda x: x[0]))

        return self.props_dict

    def _arg2start_end(self, arg_dict):
        '''Converts flat tree format (`05`) to Start End format (`04`)

        Converts a proposition dict (index: arg) in CoNLL 2005 format 
        to CoNLL 2004.

        Keyword Arguments:
            prop_dict {[type]} -- [description] (default: {None})
        '''
        propositions = {idx: self.db['P'][idx] for idx in arg_dict}

        # br.propbankbr_arg2se(arg_list) expects a list or tuples
        # first element of the tuple is `PRED`
        # second elemnt of the tuple is `ARG`
        # propositions are separated by None
        arg_list = []
        prev_prop = None
        for idx, prop in propositions.items():
            if prop != prev_prop and prev_prop is not None:
                arg_list.append(None)
            pred_ = self.idx2lex['PRED'][self.db['PRED'][idx]]
            arg_list.append((pred_, arg_dict[idx]))
            prev_prop = prop

        SE = br.propbankbr_arg2se(arg_list)
        se_list = [se[1] for se in SE if se is not None]
        # Converts SE into dictonary of propositions
        # zip_list = [arg_list, se_list]

        zip_list = zip(arg_dict.keys(), se_list)
        se_dict = OrderedDict(sorted(zip_list, key=lambda x: x[0]))
        return se_dict
