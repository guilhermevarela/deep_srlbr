'''
    Created on Apr 11, 2018
        @author: Varela

    Implements a thin wrapper over liblinear

'''
import sys
sys.path.append('../svmlib')

import liblinearutils as lin


class SVM(object):
    _svm = None

    @classmethod
    def read(cls, svmproblem_path):
        Y, X = lin.svm_read_problem(svmproblem_path)
        return Y, X

    def fit(self, X, Y, argsstr):
        self._svm = lin.train(Y, X, argsstr)

    def predict(self, X, Y):
        Yhat, acc, metrics = lin.predict(Y, X, self._svm)
        return Yhat, acc, metrics
