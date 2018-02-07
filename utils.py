'''
Created on Feb 07, 2018
	
	@author: Varela
	
	Common machine learning functions for sequence examples on the form 
		[BATCH_SIZE, TIME_MAX, FEATURE_SIZE]
'''

import tensorflow as tf

def cross_entropy(probs, targets):
  # Compute cross entropy for each sentence
  xentropy = tf.cast(targets, tf.float32) * tf.log(probs)
  xentropy = -tf.reduce_sum(xentropy, 2)
  mask = tf.sign(tf.reduce_max(tf.abs(targets), 2)) 
  mask = tf.cast(mask, tf.float32)
  xentropy *= mask
  # Average over actual sequence lengths.
  xentropy = tf.reduce_sum(xentropy, 1)
  xentropy /= tf.reduce_sum(mask, 1)
  return tf.reduce_mean(xentropy)

def error_rate(probs, targets, sequence_length):
  mistakes = tf.not_equal(
      tf.argmax(targets, 2), tf.argmax(probs, 2))
  mistakes = tf.cast(mistakes, tf.float32)
  mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
  mask = tf.cast(mask, tf.float32)
  mistakes *= mask
  # Average over actual sequence lengths.
  mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
  mistakes /= tf.cast(sequence_length, tf.float32)
  return tf.reduce_mean(mistakes)