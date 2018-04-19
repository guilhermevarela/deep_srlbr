'''
    Created on Apr 11, 2018
        @author: Varela

    Implements a thin wrapper over liblinear
    Call this script from project root

'''


import pickle
import os
import glob
import svmlib.liblinearutil as lin


class SVM(object):
    _svm = None

    @classmethod
    def read(cls, svmproblem_path):
        Y, X = lin.svm_read_problem(svmproblem_path)
        return Y, X

    def fit(self, X, Y, argstr):
        self._svm = lin.train(Y, X, argstr)

    def predict(self, X, Y):
        # return pred_labels, (ACC, MSE, SCC), pred_values
        labels, metrics, values = lin.predict(Y, X, self._svm)

        d = {
            'yhat': labels,
            'acc': metrics[0],
            'mse': metrics[1],
            'scc': metrics[2],
            'val': values
        }
        return d


class _SVMIO(object):

    @classmethod
    def read(cls, svmproblem_path):
        Y, X = lin.svm_read_problem(svmproblem_path)
        return Y, X

    @classmethod
    def dump(cls, encoding, optargs, **kwargs):
        '''
            Writes output in pickle format
        '''
        # print(kwargs)
        hparam = '_'.join(sorted(optargs.split('-')))
        hparam = hparam.replace(' ', '-')
        hparam =  encoding + hparam

        target_dir = 'outputs/svm/{:}/'.format(hparam)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        target_glob = '{:}[0-9]*'.format(target_dir)
        num_outputs = len(glob.glob(target_glob))
        target_dir += '{:02d}/'.format(num_outputs)
        os.mkdir(target_dir)

        for key, value in kwargs.items():
            picklename = '{:}{:}.pickle'.format(target_dir, key)
            with open(picklename, '+wb') as f:
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    svm = SVM()

    # optargs = '-c 4'
    # optargs = ('-s ')
    # encoding = 'emb'
    encoding = 'glo'
    print('Loading train set ...')
    # input_path = 'datasets/svms/{:}/train_LEMMA_glove_s50.svm'.format(encoding)
    input_path = 'datasets/svms/{:}/train.svm'.format(encoding)
    Ytrain, Xtrain = _SVMIO.read(input_path)
    print('Loading train set ... done')

    print('Loading validation set ...')
    input_path = 'datasets/svms/{:}/valid.svm'.format(encoding)
    Yvalid, Xvalid = _SVMIO.read(input_path)
    print('Loading validation set ... done')

    #(1, 3, 4, 5, 6, 7)
    for s in [7]:
        # optargs = '-s {:} -v 10'.format(s)
        optargs = '-s {:}'.format(s)
        print('Training ... with_optargs({:})'.format(optargs))
        svm.fit(Xtrain, Ytrain, optargs)
        print('Training ... done')

        keys = ('y_hat', 'acc', 'mse', 'scc')
        print('Insample prediction ...')
        train_d = svm.predict(Xtrain, Ytrain)
        print('Insample prediction ... done')


        print('Outsample prediction ...')
        valid_d = svm.predict(Xvalid, Yvalid)
        print('Outsample prediction ... done')


        _SVMIO.dump(encoding, optargs, train=train_d, valid=valid_d)
