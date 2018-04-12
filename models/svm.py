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
    def dump(cmd, **kwargs):
        '''
            Writes output in pickle format
        '''
        target_dir = 'outputs/svm/{:}/'.format(cmd)
        if not os.path.exists(target_dir):
            os.mknod(target_dir)

        target_glob = '{:}[0-9]'.format(target_dir)
        num_outputs = len(glob.glob(target_glob))
        target_dir += '{:2d}/'.format(num_outputs)

        for _arg, _val in kwargs:
            filename = '{:}{:}'.format(target_dir, _arg)
            with open(filename, 'wb') as f:
                pickle.dump(_val, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    svm = SVM()
    cmdstr = '-c 4'
    Y, X = _SVMIO.read('datasets/svms/emb/valid_LEMMA_glove_s50.svm')
    svm.fit(X, Y, cmdstr)
    Yhat, accuracy, metrics = svm.predict(X, Y)
    _SVMIO.read(Yhat=Yhat, accuracy=accuracy, metrics=metrics)
