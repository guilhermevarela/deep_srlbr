import re
from subprocess import Popen, PIPE, STDOUT

from models.propbank_encoder import PropbankEncoder
import config

PEARL_SRLEVAL_PATH = 'srlconll-1.1/bin/srl-eval.pl'

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


if __name__ == '__main__':
    encoder_path = '{:}{:}.pickle'.format(config.INPUT_DIR, 'deep_glo50')
    propbank_encoder = PropbankEncoder.recover(encoder_path)


    # def iterator(self, ds_type, filter_columns=['P', 'T'], encoding='EMB'):
    columns_ = ['INDEX', 'P', 'FORM', 'GPOS', 'PRED','ARG']
    iter_ = propbank_encoder.iterator('valid', filter_columns=columns_, encoding='CAT')
    first = True

    prev_tag_ = '*'
    prev_prop_ = -1

    baseline_list = []
    gold_list = []
    # for index_, prop_, form_, pred_, arg_ in iter_:
    for index_, dict_ in iter_:
        # import code; code.interact(local=dict(globals(), **locals()))
        prop_ = dict_['P']
        form_ = dict_['FORM']
        pred_ = dict_['PRED']
        arg_  = dict_['ARG']
        gpos_ = dict_['GPOS']

        if not (first or prop_ == prev_prop_):
            baseline_list.append(None)
            gold_list.append(None)

            prev_tag_ = '*'

        tag_ = '*'
        # VERB RULE
        if not pred_ == '-':
            # if prev_form_ == 'se':
            #     tag_ = '(C-V*)'
            #     prev_tag_ = '(V*)'
            # else:
            tag_ = '(V*)'
        elif prev_tag_ == '(V*)' and form_ == 'se':
            tag_ = '(C-V*)'

        # NEGATION RULE
        if form_ == 'não':
            tag_ = '(AM-NEG*)'


        # if not first:
        gold_list.append((pred_, arg_))
        baseline_list.append((pred_, tag_))

        prev_tag_ = tag_
        prev_prop_ = prop_
        first = False

    evaluate(gold_list, baseline_list, verbose=True)













