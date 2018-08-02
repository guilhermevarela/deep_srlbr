'''
Created on Jan 25, 2018
    @author: Varela

    Generates and reads tfrecords
        * Generates train, valid, test datasets
        * Provides input_with_embeddings_fn that acts as a feeder


    
    2018-02-21: added input_with_embeddings_fn
    2018-02-26: added input_labels to input_with_embeddings_fn
    2018-03-02: added input_validation and input_train
'''
import sys
ROOT_DIR = '/'.join(sys.path[0].split('/')[:-2]) # UGLY PROBLEM FIXES TO LOCAL ROOT --> import config
sys.path.append(ROOT_DIR)
sys.path.append('./models/')

import config as conf
import numpy as np


#Uncomment if launched from /datasets
# from propbank_encoder import PropbankEncoder
# from models import PropbankEncoder

import tensorflow as tf
from collections import defaultdict

TF_SEQUENCE_FEATURES = {
    key: tf.VarLenFeature(tf.int64)
    for key in conf.SEQUENCE_FEATURES
}

TF_CONTEXT_FEATURES = {
    'L': tf.FixedLenFeature([], tf.int64)
}

TF_SEQUENCE_FEATURES_V2 = {
    key: tf.VarLenFeature(tf.int64)
    for key in conf.CATEGORICAL_FEATURES
}
TF_SEQUENCE_FEATURES_V2.update({
    key: tf.VarLenFeature(tf.float32)
    for key in conf.EMBEDDED_FEATURES
})


def get_test(input_labels, output_label, embeddings='glo50'):
    return tfrecords_extract('test', input_labels, output_label, embeddings)


def get_valid(input_labels, output_label, embeddings='glo50'):
    return tfrecords_extract('valid', input_labels, output_label, embeddings)


def get_train(input_labels, output_label, embeddings='glo50'):
    return tfrecords_extract('train', input_labels, output_label, embeddings)


def get_tfrecord(ds_type, embeddings):
    if embeddings:
        _preffix = '{:}/db{:}_{:}.tfrecords'
        return _preffix.format(conf.INPUT_DIR, ds_type, embeddings)
    else:
        _preffix = '{:}/db{:}.tfrecords'
        return _preffix.format(conf.INPUT_DIR, ds_type)


def get_size(ds_type):
    if ds_type in ['train']:
        dataset_size = conf.DATASET_TRAIN_SIZE
    elif ds_type in ['test']:
        dataset_size = conf.DATASET_TEST_SIZE
    elif ds_type in ['valid']:
        dataset_size = conf.DATASET_VALID_SIZE
    return dataset_size


def tfrecords_extract(ds_type, input_labels, output_label, embeddings):
    '''Converts the contents of ds_type (train, valid, test) into array

    Acts as a wrapper for the function tsr2npy

    Arguments:
        ds_type {str} -- train, valid, test
        input_labels {list<str>} -- labels from the db columns 
        output_label {str} -- target_label (T, IOB)
        embeddings {str]} -- emebddings name (glo50, wan50, wrd50)

    Returns:
        inputs  {numpy.array} -- a 3D array NUM_RECORDS X MAX_TIME X FEATURE_SIZE
        targets {numpy.array} -- a 3D array NUM_RECORDS X MAX_TIME X TARGET_SIZE
        lengths {list<int>} -- a list holding the lengths for every proposition
        others {numpy.array} -- a 3D array representing the indexes


    Raises:
        ValueError -- validation for database type 
    '''
    if not(ds_type in ['train', 'valid', 'test']):
        buff = 'ds_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(ds_type)
        raise ValueError(buff)
    else:
        dataset_path = get_tfrecord(ds_type, embeddings)
        dataset_size = get_size(ds_type)

    inputs, targets, lengths, others = tsr2npy(
        dataset_path,
        dataset_size,
        input_labels,
        output_label,
        msg='input_{:}: done converting {:} set to numpy'.format(ds_type, ds_type)
    )

    return inputs, targets, lengths, others


def tsr2npy(dataset_path, dataset_size,
            input_labels, output_label, msg='tensor2numpy: done'):
    '''Converts .tfrecords binary format to numpy.array


    Arguments:
        dataset_path {str} -- a str path
        dataset_size {int} -- total number of propositions
        input_labels {list<str>} -- labels from the db columns 
        output_label {str} -- target_label (T, IOB)

    Returns:
        inputs  {numpy.array} -- a 3D array size
            NUM_RECORDS X MAX_TIME X FEATURE_SIZE
        targets {numpy.array} -- a 3D array size
            NUM_RECORDS X MAX_TIME X TARGET_SIZE
        lengths {numpy.array} -- a 1D array size
            lengths for every proposition
        descriptors {list<str>} -- a 3D array NUM_RECORDS X MAX_TIME 
            index of tokens

    Raises:
        ValueError -- validation for database type 
    '''
    with tf.name_scope('pipeline'):
        X, T, L, D = input_fn(
            [dataset_path], dataset_size, 1,
            input_labels, output_label, shuffle=False
        )

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # This first loop instanciates validation set
        try:
            while not coord.should_stop():
                inputs, targets, times, descriptors = session.run([X, T, L, D])

        except tf.errors.OutOfRangeError:
            print(msg)

        finally:
            # When done, ask threads to stop
            coord.request_stop()
            coord.join(threads)

    return inputs, targets, times, descriptors


# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, batch_size, num_epochs,
             input_labels, output_label, shuffle=True):
    '''[summary]

    [description]

    Arguments:
        filenames {list<str>} -- list containing tfrecord file names
        batch_size {int} -- integer containing batch size
        num_epochs {int} -- integer representing the total number 
            of iterations on filenames.
        input_labels {[type]} -- list containing the feature fields from .csv file
        output_label {[type]} -- column

    Keyword Arguments:
        shuffle {bool} -- If true will shuffle the proposition for 
            each epoch (default: {True})

    Returns:
        X_batch {numpy.array} -- a 3D array size
            NUM_RECORDS X MAX_TIME X FEATURE_SIZE
        T_batch  {numpy.array} -- a 3D array size
            NUM_RECORDS X MAX_TIME X TARGET_SIZE                        .:
        L_batch  {numpy.array} -- a 1D array size
            lengths for every proposition
        D_batch  {list<str>} -- a 3D array NUM_RECORDS X MAX_TIME 
            index of tokens
    '''
    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=shuffle
    )
    context_features, sequence_features = _read_and_decode(filename_queue)

    X, T, L, D = _process(
        context_features,
        sequence_features,
        input_labels,
        output_label
    )

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size


    X_batch, T_batch, L_batch, D_batch = tf.train.batch(
        [X, T, L, D],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True
    )
    return X_batch, T_batch, L_batch, D_batch


def _process(context_features, sequence_features,
             input_labels, output_label):
    '''Maps context_features and sequence_features making embedding replacement as necessary

        args:
            context_features   .: protobuffer containing features hold constant for example 
            sequence_features  .: protobuffer containing features that change wrt time 
            input_labels       .: list of sequence features to be used as inputs
            embeddings         .: matrix [EMBEDDING_SIZE, VOCAB_SIZE] containing pre computed word embeddings
            klass_size,        .: int number of output classes

        returns:    
            X                                   .: 
            T                       .: 
            L                               .: 
            D               .:

            2018-02-26: input_labels introduced now client may select input fields
                (before it was hard coded and alll features from .csv where used)
    '''
    sequence_inputs = []
    sequence_descriptors = []


    # Fetch only context variable the length of the proposition
    L = context_features['L']

    print('processing:{:}'.format(sequence_features.keys()))
    for key in TF_SEQUENCE_FEATURES_V2:

        dense_tensor = tf.sparse_tensor_to_dense(sequence_features[key])

        dense_tensor1 = tf.cast(dense_tensor, tf.float32)

        if key in input_labels:
            sequence_inputs.append(dense_tensor1)
        elif key in [output_label]:
            print('target is', output_label)
            T = dense_tensor1
        elif key in ['INDEX']:
            print('descriptors: {:}'.format(key))
            sequence_descriptors.append(dense_tensor1)

    X = tf.concat(sequence_inputs, 1)
    D = tf.concat(sequence_descriptors, 1)
    return X, T, L, D


def _read_and_decode(filename_queue):
    '''
        Decodes a serialized .tfrecords containing sequences
        args
            filename_queue.: list of strings containing file names which are added to queue

        returns
            context_features.: features that are held constant thru sequence ex: time, sequence id

            sequence_features.: features that are held variable thru sequence ex: word_idx

    '''
    print(TF_SEQUENCE_FEATURES_V2)
    reader= tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=TF_CONTEXT_FEATURES,
        sequence_features=TF_SEQUENCE_FEATURES_V2
    )

    return context_features, sequence_features

def read_sequence_example(file_path):
    reader = tf.python_io.tf_record_iterator(file_path)
    for sequence_example_string in reader:
        ex = tf.train.SequenceExample().FromString(sequence_example_string)


def make_feature_list(columns_dict, columns_config):
    '''Returns a SequenceExample for the given inputs and labels.

    args:
        columns_dict    .: dict<str, object>
                            keys: are the column labels
                            values: are either list of floats, list of integer or integer

        columns_config  .: in memory representation of datasets/schemas

    returns:
        feature_dict    .: dict<>
    '''
    feature_dict = {}
    supported_types = ('choice', 'int', 'text')
    for label_, value_ in columns_dict.items():
        # makes approximate comparison
        # label_ = label_.split('_')[0]

        config_dict = columns_config[label_]

        if config_dict['type'] in ('bool', 'int'):
            # the representation is an integer list
            feature_dict[label_] = tf.train.FeatureList(
                feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
                         for value in value_])
        elif config_dict['type'] in ('choice',):
            # the representation is an integer list or list
            feature_dict[label_] = tf.train.FeatureList(
                feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=sublist))
                         for sublist in value_])
        elif config_dict['type'] in ('text',):
            # the representation is float array
            feature_dict[label_] = tf.train.FeatureList(
                feature=[tf.train.Feature(float_list=tf.train.FloatList(value=sublist))
                         for sublist in value_])
        else:
            _params = (supported_types, config_dict['type'])
            _msg = 'type must be in {:} and got {:}'
            raise ValueError(_msg.format(*_params))
    return feature_dict


def tfrecords_builder(propbank_iter, dataset_type,
                      column_config_dict, suffix='glo50'):
    ''' Iterates within propbank and saves records

        ref https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/feature.proto
            https://planspace.org/20170323-tfrecords_for_humans/
    '''
    if not(dataset_type in ['train', 'valid', 'test']):
        buff = 'dataset_type must be \'train\',\'valid\' or \'test\' got \'{:}\''.format(dataset_type)
        raise ValueError(buff)
    else:
        if dataset_type in ['train']:
            total_propositions = conf.DATASET_TRAIN_SIZE
        if dataset_type in ['valid']:
            total_propositions = conf.DATASET_VALID_SIZE
        if dataset_type in ['test']:
            total_propositions = conf.DATASET_TEST_SIZE

    def message_print(num_records, num_propositions):
        _msg = 'Processing {:}\tfrecords:{:5d}\tpropositions:{:5d}\tcomplete:{:0.2f}%\r'
        _perc = 100 * float(num_propositions) / total_propositions
        _msg = _msg.format( dataset_type, num_records, num_propositions, _perc)
        sys.stdout.write(_msg)
        sys.stdout.flush()

    file_path = conf.INPUT_DIR
    file_path += 'db{:}_{:}.tfrecords'.format(dataset_type, suffix)
    with open(file_path, 'w+') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
        seqlen = 1
        refresh = True
        prev_p = -1
        num_records = 0
        num_propositions = 0
        for index, d in propbank_iter:
            if d['P'] != prev_p:
                if not refresh:
                    context = {'L': tf.train.Feature(int64_list=tf.train.Int64List(value=[seqlen]))}                    
                    feature_lists_dict = make_feature_list(feature_list_dict, column_config_dict)

                    ex = tf.train.SequenceExample(
                        context=tf.train.Features(feature=context),
                        feature_lists=tf.train.FeatureLists(feature_list=feature_lists_dict)
                    )
                    writer.write(ex.SerializeToString())
                refresh = False
                seqlen = 1
                context = {}
                feature_list_dict = defaultdict(list)
                num_propositions += 1
            else:
                seqlen += 1

            for feat, value in d.items():
                feature_list_dict[feat].append(value)

            num_records += 1
            prev_p = d['P']
            if num_propositions % 25:
                message_print(num_records, num_propositions)

        message_print(num_records, num_propositions)

        # write the one last example
        context = {'L': tf.train.Feature(int64_list=tf.train.Int64List(value=[seqlen]))}
        feature_lists_dict = make_feature_list(feature_list_dict, column_config_dict)

        ex = tf.train.SequenceExample(
            context=tf.train.Features(feature=context),
            feature_lists=tf.train.FeatureLists(feature_list=feature_lists_dict)
        )
        writer.write(ex.SerializeToString())

    writer.close()
    print('Wrote to {:} found {:} propositions'.format(f.name, num_propositions))


if __name__== '__main__':
    # Recover a propbank representation
    embeddings = 'wan50'
    bin_path = 'datasets/binaries/deep_{:}.pickle'.format(embeddings)
    propbank_encoder = PropbankEncoder.recover(bin_path)

    # Gets an iterator
    column_filters = None
    colcofig_dict = propbank_encoder.columns_config  # SEE schemas/gs.yaml
    for ds_type in ('test', 'valid', 'train'):
        tfrecords_builder(propbank_encoder.iterator(ds_type, filter_columns=column_filters), ds_type, colcofig_dict, suffix=embeddings)


    input_list = ['ID', 'FORM', 'LEMMA', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1',
                  'GPOS_CTX_P-1', 'GPOS_CTX_P+0', 'GPOS_CTX_P+1']

    TARGET = 'T'
    X_test, Y_test, L_test, D_test = get_test(input_list, TARGET, embeddings=embeddings)

    # Inspect a sequence_example
    # read_sequence_example('datasets/binaries/dbtest_wan50.tfrecords')