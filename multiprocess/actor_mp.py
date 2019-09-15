#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:45:19 2018

@author: lihaoruo
"""
import numpy as np
import tensorflow as tf
from model import network
import random


def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

def update_target_graph(from_vars,to_scope):
    op_holder = []
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

class Actor():
    def __init__(self,sess,env,name,s_size,a_size,queues):
        self.name = "worker_" + str(name)
        self.number = name
        self.sess = sess
        self.Actor_net = network(s_size, a_size, self.name)
        self.env = env
        
        self.queue = queues[0]
        self.param_queue = queues[1]
        self.sess.run(tf.global_variables_initializer())
        
    def run(self):
        print ("Starting actor " + str(self.number))
        epsilon = 0.15
        epi_q = []
        while True:
            #print self.name
            total_steps = 0
            episode_reward = 0
            d = False
            s = self.env.reset()
            s = process_frame(s)
            if epsilon > 0.005:
                epsilon = epsilon * 0.999
            
            if not self.param_queue.empty():
                learner_params = self.param_queue.get()
                self.sess.run(update_target_graph(learner_params, self.name))
                #print 'updata parameters...'
            while not d:
                #self.env.render()
                if random.random() > epsilon:
                    a_dist = self.sess.run(self.Actor_net.Q, feed_dict={self.Actor_net.inputs:[s]})[0]
                    a = np.argmax(a_dist)
                else:
                    a = random.randint(0, 5)
                    
                s1, r, d, _ = self.env.step(a)
                if d != True:
                    s1 = process_frame(s1)
                else:
                    s1 = s
                
                #if self.queue.full():
                #    print self.name
                if total_steps % 4 == 0:
                    self.queue.put((s,a,r,s1,d))
                episode_reward += r
                s = s1
                total_steps += 1
            epi_q.append(episode_reward)
            if total_steps % 10 == 0:
                print (self.name, np.mean(epi_q[-10:]))
            
            if len(epi_q) > 20:
                epi_q.pop(0)

