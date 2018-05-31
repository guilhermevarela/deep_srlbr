'''
    Created on May 31, 2018

    @author: Varela

    refs:
        https://danijar.com/structuring-your-tensorflow-models/
'''

import tensorflow as tf
from tf.nn.rnn_cell import BasicLSTMCell
import functools


def get_unit(sz):
    return BasicLSTMCell(sz, forget_bias=1.0, state_is_tuple=True)


def lazy_property(function):
    '''
        It stores the result in a member named after the decorated function 
        (prepended with a prefix) and returns this value on any subsequent calls.        
    '''
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DataLayer(object):
    '''
      Defines the first layer for DBLSTM model
    '''
    def __init__(self):
        pass

    @lazy_property
    def prediction():
        pass


class HiddenLayer(object):
    '''
      Defines the hidden layers for model
    '''
    def __init__(self):
        pass

    @lazy_property
    def prediction():
        pass


class DBLSTM(object):
    def __init__(self):
        pass

    @lazy_property
    def prediction(self):
        pass

    @lazy_property
    def optimize(self):
        pass

    @lazy_property
    def error(self):
        pass
