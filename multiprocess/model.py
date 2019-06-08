#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:38:55 2019

@author: lihaoruo
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

#def network(s_size, a_size, scope, N=20):
class network():
     def __init__(self, s_size, a_size, scope, N=20):
         with tf.variable_scope(scope):
            quantile = 1.0 / N
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=imageIn,num_outputs=32,
                    kernel_size=[8,8],stride=[4,4],padding='VALID')
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=conv1,num_outputs=64,
                    kernel_size=[4,4],stride=[2,2],padding='VALID')
            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=conv2,num_outputs=64,
                    kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(conv3),512,activation_fn=tf.nn.relu)
    
            out = slim.fully_connected(hidden ,a_size * N,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.q = tf.reshape(out, [-1, a_size, N])
            self.Q = tf.reduce_sum(self.q * quantile, axis=2)
            
            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            #return self.Q, q, local_vars