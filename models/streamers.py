'''
    Created on Sep 20, 2018

    @author: Varela

    Provides functions that map tensorflow tfrecords
    binaries and the execution graph
'''
import tensorflow as tf

from models.properties import lazy_property
from datasets import input_with_embeddings_fn


class TfStreamerWE(object):
    '''TfStreamerWE performs data streaming for tensorflow's binary module

    TfStramerWE applies Word Embeddings at run time for `text` features
    -- `bool` and `int` features are regarded as having dim = 1
    -- ``
    '''
    def __init__(self, word_embeddings, trainable, datasets_list,
                 batch_size, epochs, input_labels, target_label,
                 dims_dict, config_dict, shuffle):
        self.we = word_embeddings
        self.trainable = trainable
        self.datasets_list = datasets_list
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_labels = input_labels
        self.target_label = target_label
        self.dims_dict = dims_dict
        self.config_dict = config_dict
        self.shuffle = shuffle

        self.stream

    @lazy_property
    def stream(self):
        '''Performs a 1-step read on datasets_list of size batch 250

        Builds the pipeline scope -- which feeds the X, T, L, I tensors

        Decorators:
            lazy_property

        Returns:
            inputs {tensor} -- features of size [BATCH_SIZE, MAX_TIME, FEATURES]
            targets {tensor} -- targets of size [BATCH_SIZE, MAX_TIME, CLASSES]
            seqlens {tensor} -- seqlens true TIME of the example rank 1 tensor
            descriptors {tensor} -- index of the example on the database rank 1 tensor
        '''
        with tf.name_scope('pipeline'):
            self.WE = tf.Variable(self.we,
                                  trainable=self.trainable, name='embeddings')

            self.inputs, self.targets, self.seqlens, self.descriptors = input_with_embeddings_fn(
                self.WE, self.datasets_list, self.batch_size,
                self.epochs, self.input_labels, self.target_label,
                self.dims_dict, self.config_dict, shuffle=True)

        return self.inputs, self.targets, self.seqlens, self.descriptors