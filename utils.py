'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Common machine learning functions for sequence examples on the form 
		X ~[BATCH_SIZE, TIME_MAX, FEATURE_SIZE]
    Y ~[BATCH_SIZE, TIME_MAX, KLASSES_SIZE]

  SEE https://danijar.com/variable-sequence-lengths-in-tensorflow/  
'''

import tensorflow as tf

def cross_entropy(probs, targets):
  '''
    Computes cross entropy considering 

    args
      probs .: 3D tensor size [batch size x max length x klasses] 
        representing the joint probability function

      targets .: 3D tensor size [batch size x max length x klasses] 
        representing the true targets

    returns
      loss .: scalar tensor
  '''
  # Compute cross entropy for each sentence
  xentropy = tf.cast(targets, tf.float32) * tf.log(probs)
  xentropy = -tf.reduce_sum(xentropy, 2)
  # mask = tf.sign(tf.reduce_max(tf.abs(targets), 2)) 
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
      probs .: 3D tensor size [batch size x max length x klasses] 
        representing the joint probability function

      targets .: 3D tensor size [batch size x max length x klasses] 
        representing the true targets

      sequence_length  .: 3D tensor size [batch size x max length x klasses] 

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

def error_rate2D(outputs, targets, sequence_length):
  '''
    Computes error rate

    args
      outputs .: 2D tensor size [batch size x max length] 
        representing the joint probability function

      targets .: 2D tensor size [batch size x max lengths] 
        representing the true targets

      sequence_length  .: 1D tensor size [batch size] 

    returns
      error .: scalar tensor 0.0 ~ 1.0
  '''
  mistakes = tf.not_equal(outputs, targets)
  mistakes = tf.cast(mistakes, tf.float32)
  
  mask = tf.sign(targets)
  mask = tf.cast(mask, tf.float32)
  mistakes *= mask
  # Average over actual sequence lengths.
  mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
  mistakes /= tf.cast(sequence_length, tf.float32)
  return tf.reduce_mean(mistakes)


def length(sequence):
  '''
    Computes true sequence length for zero pedded tensor

    args
      sequence .: 3D tensor size [batch size x max length x N] 
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
      sequence .: 3D tensor size [batch size x max length x N] 
        must be zero padded 

    returns
      mask .: 3D tensor size [batch size x max length x N] 
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

