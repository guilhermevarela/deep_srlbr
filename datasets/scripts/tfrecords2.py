'''
Created on Jan 25, 2018
    @author: Varela

    Generates and reads tfrecords
        * Generates train, valid, test datasets
        * Provides input_with_embeddings_fn that acts as a feeder
    EX.:
    # Recover a propbank representation
    > from models.propbank_encoder import PropbankEncoder
    > embeddings = 'wan50'
    > bin_path = 'datasets/binaries/1.0/deep_{:}.pickle'.format(embeddings)
    > propbank_encoder = PropbankEncoder.recover(bin_path)

    # Build a propbank iterator
    > column_filters = None
    > config_dict = propbank_encoder.columns_config
    > for ds_type in ('test', 'valid', 'train'):
    >   iterator_ = propbank_encoder.iterator(ds_type, filter_columns=column_filters)
    >   tfrecords_builder(, ds_type, config_dict, suffix=embeddings)

    # Inspect a sequence_example
    > read_sequence_example('datasets/binaries/dbtest_wan50.tfrecords')

    # Get the test dataset
    > input_list = ['ID', 'FORM', 'MARKER', 'GPOS', 'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']
    > dims_dict = propbank_encoder.columns_dimensions('EMB')

    > TARGET = 'T'
    > X_test, Y_test, L_test, D_test = get_test(input_list, TARGET,
                                                embeddings=embeddings, dimensions_dict=dims_dict)

    # Inspect a sequence_example
    > read_sequence_example('datasets/binaries/dbtest_wan50.tfrecords')

    2018-02-21: added input_with_embeddings_fn
    2018-02-26: added input_labels to input_with_embeddings_fn
    2018-03-02: added input_validation and input_train
'''
import collections
from collections import defaultdict
import sys
# import os
# sys.path.append(os.getcwd())
import tensorflow as tf
import config as conf
from utils.info import get_db_bounds

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


def get_test(embeddings, input_labels, output_labels, dims_dict,
             config_dict, embeddings_model='glo50', version='1.0'):

    return tfrecords_extract('test', embeddings, input_labels, output_labels,
                             dims_dict, config_dict, embeddings_model,
                             version=version)


def get_valid(embeddings, input_labels, output_labels, dims_dict,
              config_dict, embeddings_model='glo50', version='1.0'):

    return tfrecords_extract('valid', embeddings, input_labels, output_labels,
                             dims_dict, config_dict, embeddings_model,
                             version=version)


def get_train(embeddings, input_labels, output_labels, dims_dict,
              config_dict, embeddings_model='glo50', version='1.0'):

    return tfrecords_extract('train', embeddings, input_labels, output_labels,
                             dims_dict, config_dict, embeddings_model,
                             version=version)


def get_test2(input_labels, output_labels, dims_dict, embeddings_model='glo50', version='1.0'):
    return tfrecords_extract2('test', input_labels, output_labels,
                              embeddings_model, dims_dict, version=version)


def get_valid2(input_labels, output_labels, dims_dict, embeddings_model='glo50', version='1.0'):
    return tfrecords_extract2('valid', input_labels, output_labels,
                              embeddings_model, dims_dict, version=version)


def get_train2(input_labels, output_labels, dims_dict, embeddings_model='glo50', version='1.0'):
    return tfrecords_extract2('train', input_labels, output_labels,
                              embeddings_model, dims_dict, version=version)


def get_tfrecord(ds_type, embeddings_model, version='1.0'):
    if embeddings_model:
        _preffix = '{:}/{:}/db{:}_{:}.tfrecords'
        return _preffix.format(conf.INPUT_DIR, version, ds_type, embeddings_model)
    else:
        _preffix = '{:}/db{:}.tfrecords'
        return _preffix.format(conf.INPUT_DIR, version, ds_type)


def get_size(ds_type, version='1.0'):
    lb, ub = get_db_bounds(ds_type, version=version)
    return ub - lb


def tfrecords_extract(ds_type, embeddings, input_labels, output_labels,
                      dimensions_dict, config_dict, embeddings_model, version='1.0'):
    '''Converts the contents of ds_type (train, valid, test) into array

    Acts as a wrapper for the function tsr2npy

    Arguments:
        ds_type {str} -- train, valid, test
        input_labels {list<str>} -- labels from the db columns
        output_labels {str} -- target_label (T, IOB)
        embeddings {str]} -- emebddings name (glo50, wan50, wrd50)

    Returns:
        inputs  {numpy.array} --  NUM_RECORDS X MAX_TIME X FEATURE_SIZE
        targets {numpy.array} --  NUM_RECORDS X MAX_TIME X TARGET_SIZE
        lengths {list<int>} -- a list holding the lengths for every proposition
        others {numpy.array} -- a 3D array representing the indexes

    Raises:
        ValueError -- validation for database type
    '''
    dataset_path = get_tfrecord(ds_type, embeddings_model, version=version)
    dataset_size = get_size(ds_type, version=version)

    msg_success = 'tfrecords_extract:'
    msg_success += 'done converting {:} set to numpy'.format(ds_type)

    inputs, targets, lengths, others = tsr2npy(
        dataset_path, embeddings, dataset_size, input_labels, output_labels,
        dimensions_dict, config_dict, msg=msg_success,
    )

    return inputs, targets, lengths, others

def tfrecords_extract2(ds_type, input_labels, output_labels,
                       embeddings_model, dims_dict, version='1.0'):
    '''Converts the contents of ds_type (train, valid, test) into array
    Acts as a wrapper for the function tsr2npy
    Arguments:
        ds_type {str} -- train, valid, test
        input_labels {list<str>} -- labels from the db columns 
        output_labels {str} -- target_label (T, IOB)
        embeddings {str]} -- emebddings name (glo50, wan50, wrd50)
    Returns:
        inputs  {numpy.array} -- a 3D array NUM_RECORDS X MAX_TIME X FEATURE_SIZE
        targets {numpy.array} -- a 3D array NUM_RECORDS X MAX_TIME X TARGET_SIZE
        lengths {list<int>} -- a list holding the lengths for every proposition
        others {numpy.array} -- a 3D array representing the indexes
    Raises:
        ValueError -- validation for database type 
    '''
    dataset_path = get_tfrecord(ds_type, embeddings_model, version=version)
    dataset_size = get_size(ds_type, version=version)

    msg_success = 'tfrecords_extract:'
    msg_success += 'done converting {:} set to numpy'.format(ds_type)

    inputs, targets, lengths, others = tsr2npy2(
        dataset_path, dataset_size, input_labels, output_labels,
        dims_dict, msg=msg_success
    )

    return inputs, targets, lengths, others


def tsr2npy(dataset_path, embeddings, dataset_size, input_labels, output_labels,
            dimensions_dict, config_dict, msg='tensor2numpy: done'):
    '''Converts .tfrecords binary format to numpy.array

    Arguments:
        dataset_path {str} -- a str path
        dataset_size {int} -- total number of propositions
        input_labels {list<str>} -- labels from the db columns
        output_labels {str} -- target_label (T, IOB)

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

        EMBS = tf.Variable(embeddings, trainable=False, name='embeddings')

        X, T, L, D = input_with_embeddings_fn(
            EMBS, [dataset_path], dataset_size, 1, input_labels, output_labels,
            dimensions_dict, config_dict, shuffle=False
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


def tsr2npy2(dataset_path, dataset_size, input_labels,
             output_labels, dims_dict, msg='tensor2numpy: done'):
    '''Converts .tfrecords binary format to numpy.array
    Arguments:
        dataset_path {str} -- a str path
        dataset_size {int} -- total number of propositions
        input_labels {list<str>} -- labels from the db columns 
        output_labels {str} -- target_label (T, IOB)
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
            [dataset_path], dataset_size, 1, input_labels,
            output_labels, dims_dict, shuffle=False
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


def input_fn(filenames, batch_size, num_epochs,
             input_labels, output_labels, dims_dict={}, shuffle=True):
    '''Provides I/O from protobufs to numpy arrays

    https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data

    Arguments:
        filenames {list<str>} -- list containing tfrecord file names
        batch_size {int} -- integer containing batch size
        num_epochs {int} -- integer representing the total number 
            of iterations on filenames.
        input_labels {[type]} -- list containing the feature fields from .csv file
        output_labels {[type]} -- column

    Keyword Arguments:
        shuffle {bool} -- If true will shuffle the proposition for 
            each epoch (default: {True})
        dim_dict {dict} -- provides the dimensions for each feature

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

    sequence_labels = input_labels + output_labels

    context_features, sequence_features = _read_and_decode2(
        filename_queue, dims_dict, sequence_labels=sequence_labels)

    X, T, L, D = _protobuf_process(
        context_features,
        sequence_features,
        input_labels,
        output_labels,
        dims_dict
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


def _protobuf_process(
        context_features, sequence_features, input_labels, output_labels, dims_dict):
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
    sequence_outputs = []
    sequence_descriptors = []

    # Fetch only context variable the length of the proposition
    L = context_features['L']    
    # Each output has a maximum pad size
    maxout_sz = max([dims_dict[lbl] for lbl in output_labels])


    labels_list = list(input_labels + output_labels)
    labels_list.append('INDEX')
    for key in labels_list:
        v = tf.cast(sequence_features[key], tf.float32)

        if key in input_labels:
            sequence_inputs.append(v)
        elif key in output_labels:
            # [0, 0] -- top, bottom [0, x] -- left right
            pad_sz = maxout_sz - dims_dict[key]
            paddings = tf.constant([[0, 0], [0, pad_sz]])
            # fills v with the right amounts of zero in orde
            v = tf.pad(v, paddings, 'CONSTANT')
            sequence_outputs.append(v)
        elif key in ['INDEX']:
            sequence_descriptors.append(v)

    X = tf.concat(sequence_inputs, 1)
    # Each target is stacked on top of each other
    if len(sequence_outputs) == 1:
        T = sequence_outputs[0]
    else:
        T = tf.stack(sequence_outputs, axis=2)
    D = tf.concat(sequence_descriptors, 1)
    return X, T, L, D


def _read_and_decode(filename_queue, dimensions_dict,
                     sequence_labels=TF_SEQUENCE_FEATURES_V2):
    '''
        Decodes a serialized .tfrecords containing sequences
        args
            filename_queue.: list of strings containing file names which are added to queue

        returns
            context_features.: features that are held constant thru sequence ex: time, sequence id

            sequence_features.: features that are held variable thru sequence ex: word_idx
    ref .: https://github.com/tensorflow/tensorflow/issues/976
    '''

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    def make_feature(key, dtype):
        return tf.FixedLenSequenceFeature(dimensions_dict[key], dtype)

    if dimensions_dict:
        sequence_features = {
            key: make_feature(key, tf.int64)
            for key in conf.CATEGORICAL_FEATURES if key in dimensions_dict
        }
        sequence_features.update({
            key: make_feature(key, tf.int64)
            for key in conf.EMBEDDED_FEATURES if key in dimensions_dict
        })

    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=TF_CONTEXT_FEATURES,
        sequence_features=sequence_features
    )

    return context_features, sequence_features


def _read_and_decode2(filename_queue, dimensions_dict,
                     sequence_labels=TF_SEQUENCE_FEATURES_V2):
    '''
        Decodes a serialized .tfrecords containing sequences
        args
            filename_queue.: list of strings containing file names which are added to queue

        returns
            context_features.: features that are held constant thru sequence ex: time, sequence id

            sequence_features.: features that are held variable thru sequence ex: word_idx
    ref .: https://github.com/tensorflow/tensorflow/issues/976
    '''

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    def make_feature(key, dtype):
        return tf.FixedLenSequenceFeature(dimensions_dict[key], dtype)

    sequence_features = {
        key: make_feature(key, tf.int64)
        for key in conf.CATEGORICAL_FEATURES if key in dimensions_dict
    }
    sequence_features.update({
        key: make_feature(key, tf.float32)
        for key in conf.EMBEDDED_FEATURES if key in dimensions_dict
    })

    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=TF_CONTEXT_FEATURES,
        sequence_features=sequence_features
    )

    return context_features, sequence_features


def read_sequence_example(file_path):
    reader = tf.python_io.tf_record_iterator(file_path)
    for sequence_example_string in reader:
        ex = tf.train.SequenceExample().FromString(sequence_example_string)
        return ex


def input_with_embeddings_fn(
    EMBS, filenames, batch_size, num_epochs, input_labels,
    output_labels, dimensions_dict, config_dict, shuffle=True):
    '''Provides I/O from protobufs to numpy arrays

    https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data

    Arguments:
        filenames {list<str>} -- list containing tfrecord file names
        batch_size {int} -- integer containing batch size
        num_epochs {int} -- integer representing the total number 
            of iterations on filenames.
        input_labels {[type]} -- list containing the feature fields from .csv file
        output_labels {[type]} -- column

    Keyword Arguments:
        shuffle {bool} -- If true will shuffle the proposition for 
            each epoch (default: {True})
        dim_dict {dict} -- provides the dimensions for each feature

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
        filenames, num_epochs=num_epochs, shuffle=shuffle
    )
    sequence_labels = list(input_labels + output_labels)

    context_features, sequence_features = _read_and_decode(
        filename_queue, dimensions_dict, sequence_labels=sequence_labels)

    X, T, L, D = _protobuf_with_embeddings_process(
        EMBS,
        context_features,
        sequence_features,
        input_labels,
        output_labels,
        config_dict
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



def _protobuf_with_embeddings_process(
        EMBS, context_features, sequence_features, input_labels, output_labels, config_dict):
    '''Maps context_features and sequence_features making embedding replacement as necessary
        
        https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation

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
    sequence_outputs = []


    # Fetch only context variable the length of the proposition
    L = context_features['L']
    k = len(sequence_labels)

    labels_list = list(input_labels + output_labels)
    labels_list.append('INDEX')
    for key in labels_list:
        ind = sequence_features[key]

        if config_dict[key]['type'] == 'text':
            ind = tf.nn.embedding_lookup(EMBS, ind)
            ind = tf.squeeze(ind, axis=1)
        else:
            ind = tf.cast(sequence_features[key], tf.float32)

        if key in input_labels:
            sequence_inputs.append(ind)
        elif key in output_labels:
            sequence_outputs.append(ind)
        elif key in ['INDEX']:
            sequence_descriptors.append(ind)

    X = tf.concat(sequence_inputs, 1)
    T = tf.concat(sequence_outputs, 1)
    D = tf.concat(sequence_descriptors, 1)
    return X, T, L, D


# def make_feature_list(columns_dict, columns_config, encoding):
def make_feature_list(column_dict, embs_model):
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
    cnf_dict = conf.get_config(embs_model)
    encoding = conf.DATA_ENCODING
    try:
        def kwargs_int64(value):
            d = {}
            if isinstance(value, collections.Iterable):
                d['int64_list'] = tf.train.Int64List(value=value)
            else:
                d['int64_list'] = tf.train.Int64List(value=[value])
            return d

        def kwargs_float(value):
            d = {}
            d['float_list'] = tf.train.FloatList(value=value)
            return d

        def kwargs_list(label_value, label_type):
            d = {}

            if label_type in ('bool', 'int', 'choice') or (encoding not in ('EMB',)):
                d['feature'] = [tf.train.Feature(**kwargs_int64(v)) for v in value]

            elif label_type in ('text',) and encoding in ('EMB',):
                d['feature'] = [tf.train.Feature(**kwargs_float(v)) for v in value]

            else:
                params_ = (supported_types, label_type)
                err_ = 'type must be in {:} and got {:}'.format(*params_)
                raise ValueError(err_)
            return d

        for label, value in column_dict.items():

            label_type = cnf_dict[label]['type']

            kwargs = kwargs_list(value, label_type)

            feature_dict[label] = tf.train.FeatureList(**kwargs)
    except TypeError as exc:
        import code; code.interact(local=dict(globals(), **locals()))

    return feature_dict


# def tfrecords_builder(propbank_iter, dataset_type, column_config_dict,
#                       encoding='EMB', suffix='glo50', version='1.0'):
def tfrecords_builder(propbank_iter, dataset_type, embs_model='glo50', version='1.0'):
    ''' Iterates within propbank and saves records

        ref https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/feature.proto
            https://planspace.org/20170323-tfrecords_for_humans/
    '''
    total_propositions = get_size(dataset_type, version=version)

    def message_print(num_records, num_propositions):
        _msg = 'Processing {:}\tfrecords:{:5d}\t'
        _msg += 'propositions:{:5d}\tcomplete:{:0.2f}%\r'
        _perc = 100 * float(num_propositions) / total_propositions
        _msg = _msg.format(dataset_type, num_records, num_propositions, _perc)
        sys.stdout.write(_msg)
        sys.stdout.flush()

    def kwargs_int64(value):
        d = {}
        if isinstance(value, collections.Iterable):
            d['int64_list'] = tf.train.Int64List(value=value)
        else:
            d['int64_list'] = tf.train.Int64List(value=[value])
        return d

    def kwargs_context(value):
        d = {}
        d['L'] = tf.train.Feature(**kwargs_int64(value))
        return d

    def kwargs_sequence(feature_dict):
        d = {}
        # f = make_feature_list(feature_dict, column_config_dict, encoding)
        f = make_feature_list(feature_dict, embs_model)
        d['feature_list'] = f
        return d

    def make_sequence_example(feature_dict, seqlen):
        context = kwargs_context(seqlen)
        sequence = kwargs_sequence(feature_list_dict)
        ex = tf.train.SequenceExample(
            context=tf.train.Features(feature=context),
            feature_lists=tf.train.FeatureLists(**sequence)
        )
        return ex

    file_path = conf.INPUT_DIR
    file_path += '{:}/'.format(version)
    file_path += 'db{:}_{:}.tfrecords'.format(dataset_type, embs_model)
    feature_list_dict = defaultdict(list)

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
                    ex = make_sequence_example(feature_list_dict, seqlen)
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

        ex = make_sequence_example(feature_list_dict, seqlen)
        writer.write(ex.SerializeToString())

    writer.close()
    print('Wrote to {:} found {:} propositions'.format(f.name, num_propositions))
