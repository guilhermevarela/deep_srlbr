'''Baseline for the Semantic Role Labeling system


'''
import re
from subprocess import Popen, PIPE
from collections import namedtuple, deque

import config
CONLL_DIR = 'datasets/txts/conll/'
TRAIN_PATH = CONLL_DIR + 'PropBankBr_v1.0_Develop.conll.txt'
PEARL_SRL04_PATH = 'srlconll04/srl-eval.pl'
PEARL_SRL05_PATH = 'srlconll05/bin/srl-eval.pl'

Chunk = namedtuple('Chunk', ('role', 'init', 'finish'))


def to_conll(atuple):
    txt = '{:}'.format(atuple[0])
    txt += '\t{:}' * (len(atuple) - 1)
    txt += '\n'

    return txt.format(*atuple[1:])


def evaluate(gold_list, eval_list, verbose=True, script_version='05'):
    '''[summary]

    Runs the CoNLL script

    Arguments:
        gold_list {[type]} -- [description]
        eval_list {[type]} -- [description]

    Keyword Arguments:
        verbose {bool} -- Prints to file and on screen (default: {True})
        script_version {str} -- CoNLL script version (default: {'05'})

    Returns:
        [type] -- [description]
    '''
    if script_version not in ('04', '05',):
        msg = 'script version {:}'.format(script_version)
        msg += 'invalid only `04` and `05` allowed'
        raise ValueError(msg)

    else:
        if script_version == '04':
            script_path = PEARL_SRL04_PATH
        else:
            script_path = PEARL_SRL05_PATH


    prefix_dir = config.BASELINE_DIR
    gold_path = '{:}train_gold.props'.format(prefix_dir)
    eval_path = '{:}train_eval.props'.format(prefix_dir)
    conll_path = '{:}eval.conll'.format(prefix_dir)

    with open(gold_path, mode='w') as f:
        for tuple_ in gold_list:
            if tuple_ is None:
                f.write('\n')
            else:
                f.write(to_conll(tuple_))

    with open(eval_path, mode='w') as f:
        for tuple_ in eval_list:
            if tuple_ is None:
                f.write('\n')
            else:
                f.write(to_conll(tuple_))

    cmd_list = ['perl', script_path, gold_path, eval_path]
    pipe = Popen(cmd_list, stdout=PIPE, stderr=PIPE)

    txt, err = pipe.communicate()
    txt = txt.decode('UTF-8')
    err = err.decode('UTF-8')

    if verbose:
        print(txt)
        with open(conll_path, mode='w') as f:
            f.write(txt)

    # overall is a summary from the list
    # is the seventh line
    lines_list = txt.split('\n')

    # get the numbers from the row
    overall_list = re.findall(r'[-+]?[0-9]*\.?[0-9]+.', lines_list[6])
    f1 = float(overall_list[-1])

    return f1


def trim(txt):
    return str(txt).rstrip().lstrip()


def invalid(txt):
    return re.sub('\n', '', txt)


def chunk_stack_process(feature_list, chunk_stack):
    time = int(feature_list[0])
    ctree = feature_list[7]

    for role_ in re.findall('\(([A-Z]*)', ctree):
        chunk_stack.append(Chunk(role=role_, init=time, finish=None))

    finished_chunks = ctree.count(')')
    c = 0
    if finished_chunks > 0:
        stack_len = len(chunk_stack)
        for i_, ck_ in enumerate(reversed(chunk_stack)):
            j_ = stack_len - (i_ + 1)
            if c < finished_chunks and ck_.finish is None:
                chunk_stack[j_] = Chunk(
                    role=ck_.role,
                    init=ck_.init,
                    finish=time
                )
                c += 1
            if c == finished_chunks:
                break


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


def update_chunk(eval_list,prop_len,  prop_ind, ck, label):
    '''Updates evaluation list to accomodate chunks


    Arguments:
        eval_list {list{tuple}} -- [description]
        prop_ind {int} -- proposition indicator
        prop_len {int} -- proposition length
        ck {Chunk{namedtuple}} -- an instance of chunk
        label {str} -- either `A0` or `A1`

    Raises:
        ValueError
    '''
    if label not in ('A0', 'A1',):
        raise ValueError('label {:} invalid'.format(label))

    if ck is not None:
        t = - (prop_len - (ck.init - 1))
        # import code; code.interact(local=dict(globals(), **locals()))
        eval_labels = list(eval_list[t])
        # try:
        eval_labels[prop_ind + 1] = '(' + label + '*'
        # except IndexError as e:
        #     import code; code.interact(local=dict(globals(), **locals()))

        eval_list[t] = tuple(eval_labels)

        t = - (prop_len - (ck.finish - 1))
        eval_labels = list(eval_list[t])
        try:
            eval_labels[prop_ind + 1] += ')'
        except IndexError as e:
            import code; code.interact(local=dict(globals(), **locals()))
        eval_list[t] = tuple(eval_labels)

        # overwirte any previously assigned labels
        for time_ in range(ck.init + 1, ck.finish):
            t = - (prop_len - (time_ - 1))
            # if eval_list[t] is None:
            #     import code; code.interact(local=dict(globals(), **locals()))
            eval_labels = list(eval_list[t])
            eval_labels[prop_ind + 1] = '*'
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

if __name__ == '__main__':
    prop_ind = 0
    sent_ind = 0
    gold_list = []
    eval_list = []
    chunk_stack = []
    predicate_dict = {}
    with open(TRAIN_PATH, mode='r') as f:
        for i, line in enumerate(f.readlines()):
            if len(line) > 1:  # Lines with scape newline \n character
                data_list = [trim(val) for val in invalid(line).split('\t')]
                feature_list = data_list[:9]
                srl_list = data_list[9:]
                num_props = len(srl_list)

                time = int(feature_list[0])
                if '(S' in feature_list[5]:
                    sent_ind += feature_list[5].count('(S')

                # Process chunk stack
                chunk_stack_process(feature_list, chunk_stack)


                gold_list.append(tuple(data_list[8:]))

                # first rule if it's the predicate mark as a verb
                eval_line = [feature_list[-1]] + ['*'] * num_props
                eval_tuple = tuple(eval_line)
                if feature_list[-1] != '-':
                    eval_line[prop_ind + 1] = '(V*)'
                    predicate_dict[prop_ind] = time
                    prop_ind += 1


                if feature_list[1] == 'não':
                    if (sent_ind > 0 and sent_ind <= num_props):
                        eval_line[sent_ind] = '(AM-NEG*)'

                eval_list.append(tuple(eval_line))
            else:
                n_chunks = len(chunk_stack)
                prop_len = time

                for prop_ind_, prop_time_ in predicate_dict.items():
                    # arg 0: look-up chunk stack `first before target verb` NP
                    ck = find_chunk0(chunk_stack, prop_time_)
                    update_chunk(eval_list, prop_len, prop_ind_, ck, 'A0')

                    # arg 1: look-up chunk stack `first after target verb` NP
                    ck = find_chunk1(chunk_stack, prop_time_)
                    update_chunk(eval_list, prop_len, prop_ind_, ck, 'A1')

                predicate_dict = {}
                prop_ind = 0
                sent_ind = 0
                gold_list.append(None)
                eval_list.append(None)
                chunk_stack = []

    evaluate(gold_list, eval_list, verbose=True)
