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
        Yhat, acc, metrics = lin.predict(Y, X, self._svm)
        return Yhat, acc, metrics


class _SVMIO(object):

    @classmethod
    def read(cls, svmproblem_path):
        Y, X = lin.svm_read_problem(svmproblem_path)
        return Y, X

    @classmethod
    def dump(cls, cmd, **kwargs):
        '''
            Writes output in pickle format
        '''
        # print(kwargs)
        hparam = '_'.join(sorted(cmd.split('-')))
        hparam = hparam.replace(' ', '-')
        hparam = hparam[1:]
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
    cmdstr = '-c 4'
    print('Loading train set ...')
    Ytrain, Xtrain = _SVMIO.read('datasets/svms/emb/train_LEMMA_glove_s50.svm')
    print('Loading train set ... done')

    print('Loading validation set ...')
    Yvalid, Xvalid = _SVMIO.read('datasets/svms/emb/valid_LEMMA_glove_s50.svm')
    print('Loading validation set ... done')

    print('Training ...')
    svm.fit(Xtrain, Ytrain, cmdstr)
    print('Training ... done')

    keys = ('y_hat', 'accuracy', 'metrics')
    print('Insample prediction ...')
    insample = svm.predict(Xtrain, Ytrain)
    print('Insample prediction ... done')
    insample = dict(zip(keys, insample))

    print('Outsample prediction ...')
    outsample = svm.predict(Xvalid, Yvalid)
    print('Outsample prediction ... done')
    outsample = dict(zip(keys, outsample))


    _SVMIO.dump(cmdstr, insample=insample, outsample=outsample)
