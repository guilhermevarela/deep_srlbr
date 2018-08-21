'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Common machine learning functions for sequence examples on the form 
		X ~[BATCH_SIZE, TIME_MAX, FEATURE_SIZE]
    Y ~[BATCH_SIZE, TIME_MAX, KLASSES_SIZE]

  SEE https://danijar.com/variable-sequence-lengths-in-tensorflow/  
'''
import config
import tensorflow as tf

def cross_entropy(probs, targets):
  '''
    Computes cross entropy considering 

    args
      probs .: 3D tensor size [batch size x MAX_TIME x klasses] 
        representing the joint probability function

      targets .: 3D tensor size [batch size x MAX_TIME x klasses] 
        representing the true targets

    returns
      loss .: scalar tensor
  '''
  # Compute cross entropy for each sentence
  xentropy = tf.cast(targets, tf.float32) * tf.log(probs)
  xentropy = -tf.reduce_sum(xentropy, 2)
  
  mask = identity(targets)
  mask = tf.cast(mask, tf.float32)
  xentropy *= mask
  # Average over actual sequence lengths.
  xentropy = tf.reduce_sum(xentropy, 1)
  xentropy /= tf.reduce_sum(mask, 1)
  return tf.reduce_mean(xentropy)

def error_rate(probs, targets, sequence_length):
  '''
    Computes error rate

    args
      probs .: 3D tensor size [batch size x MAX_TIME x klasses] 
        representing the joint probability function

      targets .: 3D tensor size [batch size x MAX_TIME x klasses] 
        representing the true targets

      sequence_length  .: 3D tensor size [batch size x MAX_TIME x klasses] 

    returns
      error .: scalar tensor 0.0 ~ 1.0
  '''
  mistakes = tf.not_equal(tf.argmax(targets, 2), tf.argmax(probs, 2))
  mistakes = tf.cast(mistakes, tf.float32)
  # mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
  mask = identity(targets)
  mask = tf.cast(mask, tf.float32)
  mistakes *= mask
  # Average over actual sequence lengths.
  mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
  mistakes /= tf.cast(sequence_length, tf.float32)
  return tf.reduce_mean(mistakes)

def error_rate2(outputs2D, targets, sequence_length):
  '''
    Computes error rate

    args
      outputs .: 2D tensor size [batch size x MAX_TIME] 
        representing the joint probability function

      targets .: 3D tensor size [batch size x MAX_TIME x KLASS_SZ] 
        representing the true targets

      sequence_length  .: 1D tensor size [batch size] 

    returns
      error .: scalar tensor 0.0 ~ 1.0
  '''
  # mask = identity(targets)
  # mask = tf.cast(mask, tf.float32)
  # maxmax= tf.reduce_max(targets, 2) * mask

  mistakes = tf.not_equal(outputs2D, targets)
  mistakes = tf.cast(mistakes, tf.float32)
  
  
  # mistakes *= mask
  # Average over actual sequence lengths.
  mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
  mistakes /= tf.cast(sequence_length, tf.float32)
  return tf.reduce_mean(mistakes)


def length(sequence):
  '''
    Computes true sequence length for zero pedded tensor

    args
      sequence .: 3D tensor size [batch size x MAX_TIME x N] 
        must be zero padded 

    returns
      sequence_length .: 1D tensor with the true length of each sequence 
  '''
  mask = identity(sequence)
  length = tf.reduce_sum(mask, 1)
  length = tf.cast(length, tf.int32)
  return length



def identity(sequence):  
  '''
    Returns a mask with ones on valid tensor entries

    args
      sequence .: 3D tensor size [batch size x MAX_TIME x N] 
        must be zero padded 

    returns
      mask .: 2D tensor size [batch size x MAX_TIME] 
        with zeros and ones 
  '''
  return tf.sign(tf.reduce_max(tf.abs(sequence), 2))


def precision(probs, targets):    
  mask= identity(targets)
  mask= tf.cast(targets, tf.float32)
  prec, _= tf.metrics.precision(tf.argmax(probs, 2), tf.argmax(targets,2), weights=tf.argmax(mask,2))
  return prec

def recall(probs, targets):    
  mask= identity(targets)
  mask= tf.cast(targets, tf.float32)
  rec, _= tf.metrics.recall(tf.argmax(probs, 2), tf.argmax(targets,2) , weights=tf.argmax(mask,2))
  return rec 


def get_db_bounds(ds_type, version='1.0'):
    '''Returns upper and lower bound proposition for dataset

    Dataset breakdowns are done by partioning of the propositions

    Arguments:
        ds_type {str} -- Dataset type this must be `train`, `valid`, `test`

    Retuns:
        bounds {tuple({int}, {int})} -- Tuple with lower and upper proposition
            for ds_type

    Raises:
        ValueError -- [description]
    '''
    ds_tuple = ('train', 'valid', 'test',)
    version_tuple = ('1.0', '1.1',)

    if not(ds_type in ds_tuple):
        _msg = 'ds_type must be in {:} got \'{:}\''
        _msg = _msg.format(ds_tuple, ds_type)
        raise ValueError(_msg)

    if not(version in version_tuple):
        _msg = 'version must be in {:} got \'{:}\''
        _msg = _msg.format(version_tuple, version)
        raise ValueError(_msg)
    else:
        size_dict = config.DATASET_PROPOSITION_DICT[version]

    lb = 1
    ub = size_dict['train']
    if ds_type == 'train':
        return (lb, ub)
    else:
      lb = ub
      ub += size_dict['valid']
      if ds_type == 'valid':
        return (lb, ub)
      else:
        lb = ub
        ub += size_dict['test']
        return (lb, ub)