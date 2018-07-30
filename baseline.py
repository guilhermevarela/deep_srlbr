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


def parse_chunk(chunk_stack, chunk_tree, t):
    i0 = 0

    # either seek or store
    store_role = False
    for i_, e_ in enumerate(chunk_tree):
        
        if store_role:
            if e_ in ('(', ')',) :
                
                chunk_type = chunk_tree[i0:i_]
                chunk_ = Chunk(chunk_type, t, None)
                chunk_stack.append(chunk_)
                i0 = i_ + 1
                store_role = e_ in ('(',)
                print('save chunk of type:', chunk_tree[i0:i_], t)         
        else:
            if e_ in ('(',):   
                print('store_role:', i_ + 1 )         
                store_role = True
                i0 = i_ + 1

        # get the next unclosed chunk and finish it
        if e_ in (')',): 
            for i_, c_ in enumerate(reversed(chunk_stack)):
                j_  = len(chunk_stack) - (i_ + 1)
                print(j_ )
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
        for chunk_ in chunk_stack 
        if t >= chunk_.init and t <= chunk_.finish ]


def baseline_rules(verb_id, negation_list, pred_list, chunk_stack):
    baseline_list = []        

    for i_, pred_ in enumerate(pred_list):
        if i_ == verb_id:
            baseline_list.append((pred_, '(V*)'))
        elif i_ in negation_list:
            # if verb_id == 25 and i_ == 24:
            #     import code; code.interact(local=dict(globals(), **locals()))
            chunks_verb =  chunk_intersection(chunk_stack, verb_id)
            chunks_neg  =  chunk_intersection(chunk_stack, i_)
            if chunks_verb[:-1] == chunks_neg[:-1]: # same parent as target verb
                baseline_list.append((pred_, '(AM-NEG*)')) 
            else:
                baseline_list.append((pred_, '*')) 
        else:
            baseline_list.append((pred_, '*')) 
    return baseline_list



if __name__ == '__main__':
    from collections import deque

    encoder_path = '{:}{:}.pickle'.format(config.INPUT_DIR, 'deep_glo50')
    propbank_encoder = PropbankEncoder.recover(encoder_path)


    
    columns_ = ['INDEX', 'ID', 'P', 'FORM', 'GPOS','CTREE', 'PRED','ARG']
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
    for index_, dict_ in iter_:
        prop_ = dict_['P']
        form_ = dict_['FORM']
        pred_ = dict_['PRED']
        arg_  = dict_['ARG']
        gpos_ = dict_['GPOS']
        ctree_ = dict_['CTREE']
        id_ = dict_['ID']

        if not (first or prop_ == prev_prop_):
            baseline_list += baseline_rules(verb_id, negation_list, pred_list, chunk_stack)

            gold_list.append(None)
            baseline_list.append(None) # file format


            chunk_stack = deque(list())
            verb_id = -1
            pred_list = []
            negation_list = []

        # SOLVE FOR CHUNKS
        # 
        # if not (first or prop_ == prev_prop_):
        #     baseline_list.append(None)
        #     gold_list.append(None)

        #     prev_tag_ = '*'
        parse_chunk(chunk_stack, ctree_, id_)

        # tag_ = '*'
        # # VERB RULE
        if not pred_ == '-':
            # if prev_form_ == 'se':
            #     tag_ = '(C-V*)'
            #     prev_tag_ = '(V*)'
            # else:
            # tag_ = '(V*)'
            verb_id =  id_ -1 
        # elif prev_tag_ == '(V*)' and form_ == 'se':
        #     tag_ = '(C-V*)'

        # # NEGATION RULE
        # if form_ == 'não' and verb_id > 0:
        if form_ == 'não':
            # import code; code.interact(local=dict(globals(), **locals()))
            negation_list.append(id_ -1 )
          
        # if not first:
        gold_list.append((pred_, arg_))
        pred_list.append(pred_)
        # baseline_list.append((pred_, tag_))

        # prev_tag_ = tag_
        prev_prop_ = prop_
        first = False
    
    baseline_list += baseline_rules(verb_id, negation_list, pred_list, chunk_stack)
    evaluate(gold_list, baseline_list, verbose=True)