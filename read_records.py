# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:20:23 2020

@author: Tu Anh dep trai
"""
import tensorflow as tf


#read tfrecords
def fd_make(len_vec):
  feature_description = {
      'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'vector': tf.io.FixedLenFeature((len_vec,), tf.float32)
  }
  return feature_description

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


def read_records(len_vec,filename):
  feature_description = fd_make(len_vec)
  raw_dataset = tf.data.TFRecordDataset(filename)
  parsed_dataset = raw_dataset.map(_parse_function)
  return parsed_dataset