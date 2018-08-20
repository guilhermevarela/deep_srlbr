'''Baseline for the Semantic Role Labeling system


'''
CONLL_DIR = 'datasets/txts/conll/'
TRAIN_PATH = CONLL_DIR + 'PropBankBr_v1.0_Develop.conll.txt'

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
    perl_cmd = ['perl', PEARL_SRLEVAL_PATH, gold_path, eval_path]
    pipe = Popen(perl_cmd, stdout=PIPE, stderr=PIPE)

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

if __name__ == '__main__'
    pass