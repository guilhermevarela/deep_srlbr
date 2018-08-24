'''
    Date: 12 Jun 2018
    Author: Guilherme Varela


    Invokes deep_srlbr scrips from command line arguments

    Usage:
    > python srl.py -help 
        Shows docs

    > python srl.py 16 16 16 16
        Estimates the DBLSTM model with the following defaults:
            * embeddings: glove of size 50.
            * predicate context: previous and posterior word from the verb features.
            * learning rate: 5 * 1e-3
            * batch size: 250
            * target: T column

'''

import argparse

from models import estimate, estimate_kfold
from models import PropbankEncoder

FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']


if __name__ == '__main__':
    #Parse descriptors
    parser = argparse.ArgumentParser(
        description='''This script uses tensorflow for multiclass classification of Semantic Roles using 
            Propbank Br built according to the Propbank guidelines. Uses Conll 2005 Shared Task pearl evaluator
            under the hood.''')

    parser.add_argument('depth', type=int, nargs='+', default=[16] * 4,
                        help='''Set of integers corresponding
                        the deep layer sizes. default: 16 16 16 16\n''')

    parser.add_argument('-embeddings', dest='embeddings', nargs=1,
                        default='glo50', choices=['glo50', 'wan50', 'wan100', 'wan300', 'wrd50'],
                        help='''Embedding abbrev.
                                and size examples: glo50, wan50.
                                Default: glo50 \n''')

    parser.add_argument('-ctx_p', dest='ctx_p', type=int, nargs=1,
                        default=1, choices=[1, 2, 3],
                        help='''Size of sliding window around predicate.
                                Default: 1\n''')

    parser.add_argument('-lr', dest='lr', type=float, nargs=1,
                        default=5 * 1e-3,
                        help='''Learning rate of the model.
                                Default: 0.005\n''')

    parser.add_argument('-batch_size', dest='batch_size', type=int, nargs=1,
                        default=250,
                        help='''Batch size.
                                Default: 250 \n''')

    parser.add_argument('-epochs', dest='epochs', type=int, nargs=1,
                        default=1000,
                        help='''Number of times to repeat training set during training.
                                Default: 1000\n''')

    parser.add_argument('-target', dest='target', nargs=1,
                        default='T', choices=['T', 'IOB', 'HEAD'],
                        help='''Target representations\n''')

    parser.add_argument('-kfold', action='store_true',
                        help='''if present performs kfold
                                optimization with 25 folds.
                                Default: False''')

    parser.add_argument('-version', type=str, dest='version',
                        nargs=1, choices=('1.0', '1.1',), default='1.0',
                        help='PropBankBr: version 1.0 or 1.1')

    args = parser.parse_args()

    input_labels = FEATURE_LABELS
    # print(args)
    if isinstance(args.ctx_p, list) and args.ctx_p[0] > 1:
        input_labels.append('FORM_CTX_P-2')
        input_labels.append('FORM_CTX_P+2')
        if args.ctx_p[0] == 3:
            input_labels.append('FORM_CTX_P-3')
            input_labels.append('FORM_CTX_P+3')

    target_label = args.target[0] if isinstance(args.target, list) else args.target
    embeddings = args.embeddings[0] if isinstance(args.embeddings, list) else args.embeddings
    learning_rate = args.lr[0] if isinstance(args.lr, list) else args.lr
    version = args.version[0] if isinstance(args.version, list) else args.version
    epochs = args.epochs[0] if isinstance(args.epochs, list) else args.epochs

    if args.kfold:
        # print(input_labels)
        # print(args.target)
        # print(args.depth)
        # print(embeddings)
        # print(args.epochs)
        # print(learning_rate)
        # print(args.batch_size)

        estimate_kfold(input_labels=input_labels, target_label=target_label,
                       hidden_layers=args.depth, embeddings=embeddings,
                       epochs=epochs, lr=learning_rate, fold=25,
                       version=version)
    else:
        # print(input_labels)
        # print(args.target)
        # print(args.depth)
        # print(embeddings)
        # print(args.epochs)
        # print(learning_rate)
        # print(args.batch_size)
        # print(args.ctx_p)

        estimate(input_labels=input_labels, target_label=target_label,
                 hidden_layers=args.depth, embeddings=embeddings,
                 epochs=epochs, lr=learning_rate,
                 batch_size=args.batch_size, version=version)
