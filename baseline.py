'''Baseline for the Semantic Role Labeling system


'''
import argparse
import re


import config
from models.evaluator_conll import evaluate
from utils.chunk import chunk_stack_process

CONLL_DIR = 'datasets/txts/conll/'
DEVELOP_PATH = CONLL_DIR + 'PropBankBr_v1.0_Develop.conll.txt'
TEST_PATH = CONLL_DIR + 'PropBankBr_v1.0_Test.conll.txt'
PEARL_SRL04_PATH = 'srlconll04/srl-eval.pl'
PEARL_SRL05_PATH = 'srlconll05/bin/srl-eval.pl'




def trim(txt):
    return str(txt).rstrip().lstrip()


def invalid(txt):
    return re.sub('\n', '', txt)



def find_chunk0(chunk_stack, ub):
    ''' Finds the candidate for argument 0

    look-up chunk stack for NP that `first before target verb`

    Arguments:
        chunk_stack {iterable{Chunk{namedtuple}} -- iterable collection of namedtuple
        ub {int} -- ID column of the verb

    Returns:
        ck {Chunk{namedtuple}} -- an instance of chunk
    '''
    for ck_ in reversed(chunk_stack):
        if ck_.role == 'NP' and ck_.finish < ub:
            return ck_

    return None


def update_argument(eval_list, prop_len, prop_ind, ck, arg_label):
    '''Updates evaluation list to accomodate chunks


    Arguments:
        eval_list {list{tuple}} -- [description]
        prop_ind {int} -- proposition indicator
        prop_len {int} -- proposition length
        ck {Chunk{namedtuple}} -- an instance of chunk
        arg_label {str} -- either `A0` or `A1`

    Raises:
        ValueError
    '''
    if arg_label not in ('A0', 'A1',):
        raise ValueError('label {:} invalid'.format(arg_label))

    if ck is not None:
        t = - (prop_len - (ck.init - 1))

        eval_labels = list(eval_list[t])
        eval_labels[prop_ind + 1] = '(' + arg_label + '*'
        eval_list[t] = tuple(eval_labels)

        t = - (prop_len - (ck.finish - 1))
        eval_labels = list(eval_list[t])

        eval_labels[prop_ind + 1] += ')'
        eval_list[t] = tuple(eval_labels)

        # overwrite any previously assigned labels
        for time_ in range(ck.init + 1, ck.finish):
            t = - (prop_len - (time_ - 1))
            eval_labels = list(eval_list[t])
            eval_labels[prop_ind + 1] = '*'
            eval_list[t] = tuple(eval_labels)


def update_neg(feature_list, eval_list, prop_len, prop_ind, ck_vp):
    '''Marks tags AM-NEG

    Updates negative tag

    Arguments:
        feature_list {list{list{str}}} -- List of lists representing sentence
        eval_list {list{tuple}} -- List of tuples representing the argument to evaluate
        prop_len {[type]} -- [description]
        prop_ind {[type]} -- [description]
        ck_vp {[type]} -- [description]
    '''
    # overwrite any previously assigned labels
    for time_ in range(ck_vp.init, ck_vp.finish):
        t = - (prop_len - (time_ - 1))
        if feature_list[t][1] == 'não':
            eval_labels = list(eval_list[t])
            eval_labels[prop_ind + 1] = '(AM-NEG*)'
            eval_list[t] = tuple(eval_labels)


def find_chunk1(chunk_stack, lb):
    ''' Finds the candidate for argument 1

    look-up chunk stack for NP that `first after target verb`

    Arguments:
        chunk_stack {iterable{Chunk{namedtuple}} -- iterable collection of namedtuple
        lb {int} -- ID column of the verb

    Returns:
        ck {Chunk{namedtuple}} -- an instance of chunk
    '''
    for ck_ in chunk_stack:
        if ck_.role == 'NP' and ck_.init > lb:
            return ck_
    return None


def filter_gold(gold_list, time, open_labels=[]):
    '''Baseline computes only a subset of tags

    Erase all tags
    * A0, A1, AM-NEG, V

    Arguments:
        gold_list {list} -- [description]
        open_labels {list} -- [description]
    '''
    if len(open_labels) == 0:
        open_labels = [False] * len(gold_list)

    filter_list = []
    for i_, g_ in enumerate(gold_list):
        if g_ in ('(A0*)', '(A1*)', '*', '(V*)', '(C-V*)', '(AM-NEG*)'):
            filter_list.append(g_)
        elif g_ in ('(A0*', '(A1*'):
            filter_list.append(g_)
            open_labels[i_] = True
        elif g_ in ('*)') and open_labels[i_]:
            filter_list.append('*)')
            open_labels[i_] = False
        else:
            filter_list.append('*')

    return filter_list, open_labels


def find_chunk_clause(chunk_stack, time):
    '''Finds the clause that contains the predicate


    Arguments:
        chunk_stack {[type]} -- [description]
        time {[type]} -- [description]
    '''
    search_ck = None
    for ck_ in chunk_stack:
        if ck_.role in ('ACL', 'FCL', 'ICL', 'CU') and \
           ck_.init <= time and ck_.finish >= time:
            if search_ck is None:
                search_ck = ck_
            else:
                # tighter the better
                if ck_.init >= search_ck.init and \
                   ck_.finish <= search_ck.finish:
                    search_ck = ck_
    return search_ck


def main(file_list, dataset, script_version='04'):
    gold_list = []
    eval_list = []
    for file_path in file_list:
        prop_ind = 0
        chunk_stack = []
        predicate_dict = {}
        open_labels = []
        feature_backlog = []
        passive_voice = [] 
        with open(file_path, mode='r') as f:
            for i, line in enumerate(f.readlines()):
                if len(line) > 1:  # Lines with scape newline \n character
                    data_list = [trim(val)
                                 for val in invalid(line).split('\t')]
                    feature_list = data_list[:9]
                    srl_list = data_list[9:]
                    num_props = len(srl_list)
                    time = int(feature_list[0])
                    open_labels = open_labels if len(open_labels) > 0 \
                        else [False] * num_props

                    passive_voice = passive_voice if len(passive_voice) > 0 \
                        else [False] * num_props

                    time_ = int(feature_list[0])
                    ctree_ = feature_list[7]
                    chunk_stack_process(time_, ctree_, chunk_stack)

                    srl_list, open_labels = filter_gold(
                        srl_list,
                        time,
                        open_labels=open_labels)
                    gold_list.append(tuple(feature_list[-1:] + srl_list))

                    eval_line = [feature_list[-1]] + ['*'] * num_props
                    if feature_list[-1] != '-':
                        eval_line[prop_ind + 1] = '(V*)'
                        predicate_dict[prop_ind] = time

                        # Passive voice
                        if feature_list[3] == 'V-PCP':
                            if len(feature_backlog) > 0:
                                if feature_backlog[-1][2] in ('ser', 'estar',):
                                    passive_voice[prop_ind] = True
                        prop_ind += 1

                    eval_list.append(tuple(eval_line))
                    feature_backlog.append(feature_list)
                else:
                    prop_len = time
                    for prop_ind_, prop_time_ in predicate_dict.items():

                        ck_clause_ = find_chunk_clause(chunk_stack, prop_time_)
                        update_neg(feature_backlog, eval_list,
                                   prop_len, prop_ind_, ck_clause_)
                        # arg 0: look-up chunk stack:
                        # `first before target verb` NP
                        ck = find_chunk0(chunk_stack, prop_time_)
                        arg = 'A1' if passive_voice[prop_ind_] else 'A0'
                        update_argument(eval_list,
                                        prop_len, prop_ind_, ck, arg)

                        # arg 1: look-up chunk stack
                        # `first after target verb` NP
                        ck = find_chunk1(chunk_stack, prop_time_)
                        arg = 'A0' if passive_voice[prop_ind_] else 'A1'
                        update_argument(eval_list, prop_len,
                                        prop_ind_, ck, arg)

                    predicate_dict = {}
                    prop_ind = 0
                    gold_list.append(None)
                    eval_list.append(None)
                    chunk_stack = []
                    open_labels = []
                    feature_backlog = []
                    passive_voice = []

    file_name = 'baseline_{:}'.format(dataset)
    evaluate(gold_list, eval_list,
             verbose=True, file_dir=config.BASELINE_DIR, file_name=file_name,
             script_version=script_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script runs a baseline using rule-based Semantic Role Labels.
                       Uses the official ConLL 2005 Shared Task pearl evaluator
                        under the hood for evaluation.''')

    parser.add_argument('--dataset', type=str, nargs=1, default='global',
                        choices=('global', 'develop', 'test'),
                        help='''String representing the database type to run
                                the baseline SRL. Default: `global`\n''')

    parser.add_argument('--script_version', type=str, nargs=1, default='04',
                        choices=('04', '05'),
                        help='''Use CoNLL 2004 or 2005 ST SRL eval script
                                Default: `04`\n''')

    args = parser.parse_args()
    dataset = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
    script_version = args.script_version[0] if isinstance(args.script_version, list) else args.script_version
    file_list = []
    if dataset in ('develop', 'global'):
        file_list.append(DEVELOP_PATH)

    if dataset in ('test', 'global'):
        file_list.append(TEST_PATH)

    main(file_list, dataset, script_version=script_version)
