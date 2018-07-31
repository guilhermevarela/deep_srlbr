import re
from subprocess import Popen, PIPE, STDOUT
from collections import namedtuple

from models.propbank_encoder import PropbankEncoder
import config

PEARL_SRLEVAL_PATH = 'srlconll-1.1/bin/srl-eval.pl'
Chunk = namedtuple('Chunk', ('role', 'init', 'finish'))


def evaluate(gold_list, eval_list, verbose=True):
    prefix_dir = config.BASELINE_DIR
    gold_path = '{:}valid_gold.props'.format(prefix_dir)
    eval_path = '{:}valid_eval.props'.format(prefix_dir)
    conll_path = '{:}eval.conll'.format(prefix_dir)
    with open(gold_path, mode='w') as f:
        for tuple_ in gold_list:
            if tuple_ is None:
                f.write('\n')
            else:
                f.write('{:}\t{:}\n'.format(*tuple_))

    with open(eval_path, mode='w') as f:
        for tuple_ in eval_list:
            if tuple_ is None:
                f.write('\n')
            else:
                f.write('{:}\t{:}\n'.format(*tuple_))

    pipe = Popen(['perl', PEARL_SRLEVAL_PATH, gold_path, eval_path], stdout=PIPE, stderr=PIPE)

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


def update_chunk_stack(chunk_stack, chunk_tree, t):
    i0 = 0

    # either seek or store
    store_role = False
    for i_, e_ in enumerate(chunk_tree):

        if store_role:
            if e_ in ('(', ')',):

                chunk_type = chunk_tree[i0:i_]
                chunk_ = Chunk(chunk_type, t, None)
                chunk_stack.append(chunk_)
                i0 = i_ + 1
                store_role = e_ in ('(',)
        else:
            if e_ in ('(',):
                store_role = True
                i0 = i_ + 1

        # get the next unclosed chunk and finish it
        if e_ in (')',):
            for i_, c_ in enumerate(reversed(chunk_stack)):
                j_ = len(chunk_stack) - (i_ + 1)
                if c_.finish is None:
                    chunk_stack[j_] = Chunk(c_.role, c_.init, t)
                    break

    if store_role:
        chunk_type = chunk_tree[i0:]
        chunk_ = Chunk(chunk_type, t, None)
        chunk_stack.append(chunk_)


def chunk_intersection(chunk_stack, t):
    return [
        chunk_
        for chunk_ in chunk_stack if chunk_enclosed(chunk_, t)]


def chunk_filter(chunk_stack, t, filter_type):

    if filter_type == 0:
        def fnc(x):
            return chunk_np(x) and chunk_lt(x, t)

    elif filter_type == 1:
        # fnc = lambda x: chunk_np(x) and chunk_gt(x, t)
        def fnc(x):
            return chunk_np(x) and chunk_gt(x, t)

    else:
        raise ValueError('Only 0 or 1 allowed')

    return [chunk_ for chunk_ in chunk_stack if fnc(chunk_)]


def chunk_enclosed(chunk, t):
    return chunk.init <= t and t <= chunk.finish


def chunk_gt(chunk, t):
    return chunk.init > t


def chunk_lt(chunk, t):
    return chunk.finish < t


def chunk_np(chunk):
    return chunk.role == 'NP*'


def baseline_rules(verb_id, cverb_id, ispassive_voice, negation_list, pred_list, chunk_stack):
    # A0 & A1 rule
    chunk_list = chunk_filter(chunk_stack, verb_id, 0)
    chunk_a0 = chunk_list[-1] if chunk_list else None

    chunk_list = chunk_filter(chunk_stack, verb_id, 1)
    chunk_a1 = chunk_list[0] if chunk_list else None

    # PASSIVE VOICE switches
    if ispassive_voice:
        chunk_a0, chunk_a1 = chunk_a1, chunk_a0

    arg_list = []
    a0 = False
    a1 = False
    for i_, pred_ in enumerate(pred_list):
        i_ += 1
        tag = '*'
        if chunk_a0 and i_ == chunk_a0.init:
            tag = '(A0*'
            a0 = True

        if chunk_a1 and i_ == chunk_a1.init:
            tag = '(A1*'
            a1 = True

        if i_ == verb_id:
            tag = '(V*)'

        if not (a1 or a0):
            if i_ in negation_list:
                chunks_verb = chunk_intersection(chunk_stack, verb_id)
                chunks_neg = chunk_intersection(chunk_stack, i_)

                # same parent as target verb
                if chunks_verb[:-1] == chunks_neg[:-1]:
                    tag = '(AM-NEG*)'

            if i_ == cverb_id:
                tag = '(C-V*)'

        if chunk_a0 and i_ == chunk_a0.finish:
            tag += ')'
            a0 = False

        if chunk_a1 and i_ == chunk_a1.finish:
            tag += ')'
            a1 = False

        arg_list.append(tag)

    return zip(pred_list, arg_list)


if __name__ == '__main__':
    from collections import deque

    encoder_path = '{:}{:}.pickle'.format(config.INPUT_DIR, 'deep_glo50')
    propbank_encoder = PropbankEncoder.recover(encoder_path)


    columns_ = ['INDEX', 'ID', 'P', 'FORM', 'LEMMA', 'GPOS', 'PRED', 'CTREE','ARG']
    iter_ = propbank_encoder.iterator('valid', filter_columns=columns_, encoding='CAT')
    first = True

    prev_tag_ = '*'
    prev_prop_ = -1

    baseline_list = []
    gold_list = []
    negation_list = []
    chunk_stack = deque(list())
    verb_id = -1

    pred_list = []
    baseline_list = []
    lemma_ = ''
    ispassive_voice = False
    cverb_id = -1
    for index_, dict_ in iter_:
        prop_ = dict_['P']
        form_ = dict_['FORM']
        lemma_ = dict_['LEMMA'] if lemma_ != 'ser' else 'ser' # once finds `ser` keeps it 
        pred_ = dict_['PRED']
        arg_  = dict_['ARG']
        gpos_ = dict_['GPOS']
        ctree_ = dict_['CTREE']
        id_ = dict_['ID']

        if not (first or prop_ == prev_prop_):
            baseline_list += baseline_rules(verb_id, cverb_id, ispassive_voice, negation_list, pred_list, chunk_stack)

            gold_list.append(None)
            baseline_list.append(None) # file format


            chunk_stack = deque(list())
            verb_id = -1
            cverb_id = -1
            pred_list = []
            negation_list = []
            ispassive_voice = False
            lemma_ = None
            prev_form_ = ''

        update_chunk_stack(chunk_stack, ctree_, id_)


        if not pred_ == '-':
            if prev_form_ == 'se':
                verb_id = id_ -1
                cverb_id = id_
            else:
                verb_id = id_

            if gpos_ == 'v-pcp' and lemma_ == 'ser':
                ispassive_voice = True
        elif verb_id == id_ - 1 and form_ == 'se':
            cverb_id = id_

        if form_ == 'não':
            negation_list.append(id_ - 1)

        gold_list.append((pred_, arg_))
        pred_list.append(pred_)

        prev_prop_ = prop_
        prev_form_ = form_
        first = False
    
    baseline_list += baseline_rules(verb_id, cverb_id, ispassive_voice, negation_list, pred_list, chunk_stack)
    evaluate(gold_list, baseline_list, verbose=True)