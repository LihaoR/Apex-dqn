#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:45:19 2018

@author: lihaoruo
"""

import tensorflow as tf
import numpy as np
from model import network
from replaymemory import ReplayMemory
import random

N = 20

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

class Learner():
    def __init__(self,sess,s_size,a_size,scope,queues,trainer):
        self.queue = queues[0]
        self.param_queue = queues[1]
        self.replaymemory = ReplayMemory(100000)
        self.sess = sess
        self.learner_net = network(s_size, a_size, scope, 20)
    
        self.q = self.learner_net.q
        self.Q = self.learner_net.Q
        
        self.actions_q = tf.placeholder(shape=[None, a_size, N], dtype=tf.float32)
        self.q_target  = tf.placeholder(shape=[None, N], dtype=tf.float32)
        self.ISWeights = tf.placeholder(shape=[None, N], dtype=tf.float32)
        
        self.q_actiona = tf.multiply(self.q, self.actions_q)
        self.q_action  = tf.reduce_sum(self.q_actiona, axis=1)
        self.u = tf.abs(self.q_target - self.q_action)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.u) * self.ISWeights, axis=1))
        
        self.local_vars = self.learner_net.local_vars#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.gradients = tf.gradients(self.loss,self.local_vars)
        #grads,self.grad_norms = tf.clip_by_norm(self.gradients,40.0)
        self.apply_grads = trainer.apply_gradients(zip(self.gradients,self.local_vars))
        self.sess.run(tf.global_variables_initializer())
            
    def run(self,gamma,s_size,a_size,batch_size,env):
        print('start learning')
        step, train1 = 0, False
        epi_q = []
        self.env = env
        while True:
            if self.queue.empty():
                pass 
            else:
                while not self.queue.empty():
                    t_error = self.queue.get()
                    step += 1
                    self.replaymemory.add(t_error)
                
            if self.param_queue.empty():
                params = self.sess.run(self.local_vars)
                self.param_queue.put(params)
            
            if step >= 10000:
                train1 = True
                step = 0
                
            if train1 == True:
                episode_buffer, tree_idx, ISWeights = self.replaymemory.sample(batch_size)
                #print 'fadsfdasfadsfa'
                episode_buffer = np.array(episode_buffer)
                #print episode_buffer
                observations      = episode_buffer[:,0]
                actions           = episode_buffer[:,1]
                rewards           = episode_buffer[:,2]
                observations_next = episode_buffer[:,3]
                dones             = episode_buffer[:,4]
                Q_target = self.sess.run(self.Q, feed_dict={self.learner_net.inputs:np.vstack(observations_next)})
                
                actions_ = np.argmax(Q_target, axis=1)
                
                action = np.zeros((batch_size, a_size))
                action_ = np.zeros((batch_size, a_size))
                for i in range(batch_size):
                    action[i][actions[i]] = 1
                    action_[i][actions_[i]] = 1
                action_now = np.zeros((batch_size, a_size, N))
                action_next = np.zeros((batch_size, a_size, N))
                for i in range(batch_size):
                    for j in range(a_size):
                        for k in range(N):
                            action_now[i][j][k] = action[i][j]
                            action_next[i][j][k] = action_[i][j]           
                q_target = self.sess.run(self.q_action, feed_dict={self.learner_net.inputs:np.vstack(observations_next),
                                                              self.actions_q:action_next})
                
                q_target_batch = []
                for i in range(len(q_target)):
                    qi = q_target[i]
                    z_target_step = []
                    for j in range(len(qi)):
                        z_target_step.append(gamma * qi[j] * (1 - dones[i]) + rewards[i])
                    q_target_batch.append(z_target_step)
                q_target_batch = np.array(q_target_batch)
                
                isweight = np.zeros((batch_size,N))
                for i in range(batch_size):
                    for j in range(N):
                        isweight[i,j] = ISWeights[i]
                feed_dict = {self.q_target:q_target_batch,
                             self.learner_net.inputs:np.vstack(observations),
                             self.actions_q:action_now,
                             self.ISWeights:isweight}
                
                l,abs_errors,_ = self.sess.run([self.loss, self.u, self.apply_grads],feed_dict=feed_dict)
                #print abs_errors
                abs_errors = np.mean(abs_errors, axis=1) + 1e-6
                
                self.replaymemory.update_priorities(tree_idx,abs_errors)
            """
            episode_reward = 0
            d = False
            s = self.env.reset()
            s = process_frame(s)

            while not d:
                #self.env.render()
                if random.random() > 0.05:
                    a_dist = self.sess.run(self.learner_net.Q, feed_dict={self.learner_net.inputs:[s]})[0]
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
                self.queue.put((s,a,r,s1,d))
                episode_reward += r
                s = s1
            epi_q.append(episode_reward)
            if step % 10 == 0:
                print 'learner', np.mean(epi_q[-10:])
            
            if len(epi_q) > 20:
                epi_q.pop(0)
            """
