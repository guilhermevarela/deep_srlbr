'''Baseline for the Semantic Role Labeling system


'''
import re
from subprocess import Popen, PIPE
from collections import namedtuple, deque

import config
CONLL_DIR = 'datasets/txts/conll/'
TRAIN_PATH = CONLL_DIR + 'PropBankBr_v1.0_Develop.conll.txt'
PEARL_SRLEVAL_PATH = 'srlconll-1.1/bin/srl-eval.pl'
Chunk = namedtuple('Chunk', ('role', 'init', 'finish'))


def to_conll(atuple):
    txt = '{:}'.format(atuple[0])
    txt += '\t{:}' * (len(atuple) - 1)
    txt += '\n'

    return txt.format(*atuple[1:])


def evaluate(gold_list, eval_list, verbose=True):
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

    cmd_list = ['perl', PEARL_SRLEVAL_PATH, gold_path, eval_path]
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
    time = feature_list[0]
    ctree = feature_list[7]

    for role_ in re.findall('\(([A-Z]*)', ctree):
        chunk_stack.append(Chunk(role=role_, init=time, finish=None))

    closes = range(ctree.count(')'))
    c = 0
    if closes > 0:
        for ck_ in reversed(chunk_stack):
            if c < closes and ck_.finish is None:
                ck_.finish = time
                c += 1
            if c == closes:
                break

def get_arg0(chunk_stack):
    for ck_ in reversed(chunk_stack):
        pass

if __name__ == '__main__':
    prop_ind = 0
    sent_ind = 0
    gold_list = []
    eval_list = []
    chunk_stack = deque(list)
    with open(TRAIN_PATH, mode='r') as f:
        for i, line in enumerate(f.readlines()):
            if len(line) > 1:  # Lines with scape newline \n character
                data_list = [trim(val) for val in invalid(line).split('\t')]
                feature_list = data_list[:9]
                srl_list = data_list[9:]
                num_props = len(srl_list)

                time = feature_list[0]
                if '(S' in feature_list[5]:
                    sent_ind += feature_list[5].count('(S')

                if '(NP' in feature_list[5]:
                    chunk_list.append(Chunk(role='NP', init=time))

                if 'NP*)' in feature_list[5]:


                gold_list.append(tuple(data_list[8:]))

                # first rule if it's the predicate mark as a verb
                eval_line = [feature_list[-1]] + ['*'] * num_props
                eval_tuple = tuple(eval_line)
                if feature_list[-1] != '-':
                    eval_line[prop_ind + 1] = '(V*)'
                    prop_ind += 1

                if feature_list[1] == 'não':
                    if (sent_ind > 0 and sent_ind <= num_props):
                        eval_line[sent_ind] = '(AM-NEG*)'


                eval_list.append(tuple(eval_line))
            else:
                prop_ind = 0
                sent_ind = 0
                gold_list.append(None)
                eval_list.append(None)
                chunk_list = deque(list)

    evaluate(gold_list, eval_list, verbose=True)
