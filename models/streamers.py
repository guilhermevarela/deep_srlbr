'''
    Created on Sep 20, 2018

    @author: Varela

    Provides functions that map tensorflow tfrecords
    binaries and the execution graph
'''
import tensorflow as tf

from models.properties import lazy_property
from datasets import input_with_embeddings_fn, input_fn
from datasets.scripts.tfrecords2 import get_train, get_valid, get_test
from datasets.scripts.tfrecords2 import get_train2, get_valid2, get_test2


class TfStreamer(object):
    '''TfStreamer performs data streaming for tensorflow's binary module

    TfStramer reads batch_size examples from tfrecrods object and
    feeds it to the computation graph

    -- `bool` and `int` features are regarded as having dim = 1
    -- `choice` features have a custom size (domain size)
    -- `text` features have embeddings size (encoded into the binary)
    '''
    def __init__(self, filenames, batch_size, num_epochs,
                 input_labels, output_labels, shuffle):

        self.filenames = filenames
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.input_labels = input_labels
        self.output_labels = output_labels
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

            self.inputs, self.targets, self.seqlens, self.descriptors = input_fn(
                self.filenames, self.batch_size, self.num_epochs,
                self.input_labels, self.output_labels, self.shuffle
            )

        return self.inputs, self.targets, self.seqlens, self.descriptors

    @classmethod
    def get_train(cls, input_labels, output_labels,
                  embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_train2(
            input_labels, output_labels, embeddings_model,
            version=version)

        return inputs, targets, seqlens, descriptors

    @classmethod
    def get_valid(cls, input_labels, output_labels,
                  embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_valid2(
            input_labels, output_labels, embeddings_model,
            version=version)

        return inputs, targets, seqlens, descriptors

    @classmethod
    def get_test(cls, input_labels, output_labels,
                 embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_test2(
            input_labels, output_labels, embeddings_model,
            version=version)

        return inputs, targets, seqlens, descriptors


class TfStreamerWE(object):
    '''TfStreamerWE performs data streaming for tensorflow's binary module

    TfStramerWE applies Word Embeddings at run time for `text` features
    -- `bool` and `int` features are regarded as having dim = 1
    -- `choice` features have a custom size
    -- `text` features that are not embedded have dim = 1
    '''
    def __init__(self, word_embeddings, trainable, datasets_list,
                 batch_size, epochs, input_labels, target_label,
                 shuffle):
        self.we = word_embeddings
        self.trainable = trainable
        self.datasets_list = datasets_list
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_labels = input_labels
        self.target_label = target_label
        self.shuffle = shuffle

        self.stream

    @classmethod
    def get_train(cls, word_embeddings, input_labels, target_label,
                  embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_train(
            word_embeddings, input_labels, target_label,
            embeddings_model, version=version
        )

        return inputs, targets, seqlens, descriptors

    @classmethod
    def get_valid(cls, word_embeddings, input_labels, target_label,
                  embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_valid(
            word_embeddings, input_labels, target_label,
            embeddings_model, version=version
        )

        return inputs, targets, seqlens, descriptors

    @classmethod
    def get_test(cls, word_embeddings, input_labels, target_label,
                 embeddings_model='glo50', version='1.0'):

        inputs, targets, seqlens, descriptors = get_test(
            word_embeddings, input_labels, target_label,
            embeddings_model, version=version
        )

        return inputs, targets, seqlens, descriptors

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
                self.epochs, self.input_labels, self.target_label, shuffle=True)

        return self.inputs, self.targets, self.seqlens, self.descriptors

