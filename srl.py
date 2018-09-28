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

from models import estimate, estimate_kfold, estimate_recover
from models import PropbankEncoder

FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']


if __name__ == '__main__':
    #Parse descriptors
    parser = argparse.ArgumentParser(
        description='''This script uses tensorflow for multiclass
            classification of Semantic Roles using Propbank Br
            built according to the Propbank guidelines.
            Uses CoNLL 2004 or 2005 Shared Task pearl evaluator
            under the hood.''')

    parser.add_argument('depth', type=int, nargs='*', default=[16] * 4,
                        help='''Set of integers corresponding
                        the deep layer sizes. default: 16 16 16 16\n''')

    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=250,
                        help='''Batch size.
                                Default: 250 \n''')

    parser.add_argument('--chunks', action='store_true',
                        help='''if present use gold standard shallow chunks
                                extracted from the Tree of Constituents.
                                Default: False''')

    parser.add_argument('--ctx_p', dest='ctx_p', type=int,
                        default=1, choices=[1, 2, 3],
                        help='''Size of sliding window around predicate.
                                Default: 1\n''')

    parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str,
                        default='',
                        help='''Checkpoint directory -- with previous
                                computed model. If this parameter is provided
                                will ignore others and load model with
                                previously set parameters.\n''')

    parser.add_argument('--embs-model', dest='embs_model', default='glo50',
                        choices=('glo50', 'wan50', 'wan100', 'wan300', 'wrd50'),
                        help='''Embedding abbrev.
                                and size examples: glo50, wan50.
                                Default: glo50 \n''')

    parser.add_argument('--epochs', dest='epochs', type=int, default=1000,
                        help='''Number of times to repeat training set during training.
                                Default: 1000\n''')

    parser.add_argument('--kfold', action='store_true',
                        help='''if present performs kfold
                                optimization with 25 folds.
                                Default: False''')

    parser.add_argument('--lr', dest='lr', type=float,
                        default=5 * 1e-3,
                        help='''Learning rate of the model.
                                Default: 0.005\n''')

    parser.add_argument('--ru', dest='ru', type=str,
                        default='BasicLSTM', choices=('BasicLSTM', 'GRU', 'LSTM'),
                        help='''Recurrent unit -- according to tensorflow.
                                Default: `BasicLSTM`\n''')

    parser.add_argument('--targets', dest='targets', default=['T'], nargs='+',
                        choices=['T', 'R', 'IOB', 'HEAD'],
                        help='''Target representations.
                        Up to two values are allowed\n''')

    parser.add_argument('--version', type=str, dest='version',
                        choices=('1.0', '1.1',), default='1.0',
                        help='PropBankBr: version 1.0 or 1.1')

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    if len(ckpt_dir) > 0:
        if ckpt_dir[-1] != '/':
            ckpt_dir += '/'
        estimate_recover(ckpt_dir)
    else:
        input_labels = FEATURE_LABELS

        ctx_p = args.ctx_p
        if ctx_p > 1:
            input_labels.append('FORM_CTX_P-2')
            input_labels.append('FORM_CTX_P+2')
            if args.ctx_p == 3:
                input_labels.append('FORM_CTX_P-3')
                input_labels.append('FORM_CTX_P+3')

        use_chunks = args.chunks
        if use_chunks:
            input_labels.append('SHALLOW_CHUNKS')

        # TODO: process target
        target_labels = args.targets

        embs_model = args.embs_model
        learning_rate = args.lr
        version = args.version
        epochs = args.epochs
        batch_size = args.batch_size
        ru = args.ru

        if args.kfold:
            estimate_kfold(input_labels=input_labels, target_labels=target_labels,
                           hidden_layers=args.depth, embeddings_model=embs_model,
                           epochs=epochs, lr=learning_rate, fold=25, ru=ru,
                           version=version, ctx_p=ctx_p, chunks=use_chunks)
        else:

            estimate(input_labels=input_labels, target_labels=target_labels,
                     hidden_layers=args.depth, embeddings_model=embs_model,
                     epochs=epochs, ru=ru, batch_size=args.batch_size,
                     version=version, ctx_p=ctx_p, lr=learning_rate,
                     chunks=use_chunks)
