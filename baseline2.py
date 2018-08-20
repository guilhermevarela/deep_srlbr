'''Baseline for the Semantic Role Labeling system


'''
CONLL_DIR = 'datasets/txts/conll/'
TRAIN_PATH = CONLL_DIR + 'PropBankBr_v1.0_Develop.conll.txt'
PEARL_SRLEVAL_PATH = 'srlconll-1.1/bin/srl-eval.pl'


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
                f.write('{:}\t{:}\n'.format(*tuple_))

    with open(eval_path, mode='w') as f:
        for tuple_ in eval_list:
            if tuple_ is None:
                f.write('\n')
            else:
                f.write('{:}\t{:}\n'.format(*tuple_))

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

if __name__ == '__main__'
    with open(TRAIN_PATH, mode='r') as f:
        for line in f.readlines():
            if len(line) > 1:  # Lines with scape newline \n character
                features_list = line.split('\t')
                import code; code.interact(local=dict(globals(), **locals()))
                # first rule if it's the predicate mark as a verb