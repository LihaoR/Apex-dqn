#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:45:19 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
import random
from replaymemory import ReplayMemory


GLOBAL_STEP = 0
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Apex_Network():
    def __init__(self,sess, s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.quantile = 1.0 / N
            self.cumulative_probabilities = (2.0 * np.arange(N) + 1) / (2.0 * N)
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.imageIn,num_outputs=32,
                                     kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv1,num_outputs=64,
                                     kernel_size=[4,4],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv2,num_outputs=64,
                                     kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv3),512,activation_fn=tf.nn.relu)

            self.policy = slim.fully_connected(hidden ,a_size * N,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden, N,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.policyk = tf.reshape(self.policy, [-1, a_size, N])
            self.valuek  = tf.reshape(self.value , [-1, 1, N])
            self.q = self.valuek + (self.policyk - tf.reduce_mean(self.policyk, axis=1, keep_dims=True))
            self.Q = tf.reduce_sum(self.q * self.quantile, axis=2)

            if scope == 'global':
                self.actions_q = tf.placeholder(shape=[None, a_size, N], dtype=tf.float32)
                self.q_target  = tf.placeholder(shape=[None, N], dtype=tf.float32)
                self.ISWeights = tf.placeholder(shape=[None, N], dtype=tf.float32)
                
                self.q_actiona = tf.multiply(self.q, self.actions_q)
                self.q_action  = tf.reduce_sum(self.q_actiona, axis=1)
                self.u = tf.abs(self.q_target - self.q_action)
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.u) * self.ISWeights, axis=1))
                
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
    def train(self,sess,gamma):
        while not coord.should_stop():
            
            episode_buffer, tree_idx, ISWeights = replaymemory.sample(batch_size)
            episode_buffer = np.array(episode_buffer)
            observations      = episode_buffer[:,0]
            actions           = episode_buffer[:,1]
            rewards           = episode_buffer[:,2]
            observations_next = episode_buffer[:,3]
            #print tree_idx
            Q_target = sess.run(self.Q, feed_dict={self.inputs:np.vstack(observations_next)})

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
            q_target = sess.run(self.q_action, feed_dict={self.inputs:np.vstack(observations_next),
                                                          self.actions_q:action_next})
            q_target_batch = []
            for i in range(len(q_target)):
                qi = q_target[i]
                z_target_step = []
                for j in range(len(qi)):
                    z_target_step.append(gamma * qi[j] + rewards[i])
                q_target_batch.append(z_target_step)
            q_target_batch = np.array(q_target_batch)
            
            isweight = np.zeros((batch_size,N))
            for i in range(batch_size):
                for j in range(N):
                    isweight[i,j] = ISWeights[i]
            feed_dict = {self.q_target:q_target_batch,
                         self.inputs:np.vstack(observations),
                         self.actions_q:action_now,
                         self.ISWeights:isweight}
            l,abs_errors,_ = sess.run([self.loss, self.u, self.apply_grads],feed_dict=feed_dict)
            #print abs_errors
            abs_errors = np.mean(abs_errors, axis=1) + 1e-6
            replaymemory.update_priorities(tree_idx,abs_errors)
            
            UPDATE_EVENT.clear()
            ROLLING_EVENT.set()

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes,lock):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.lock = lock

        self.local_Apex = Apex_Network(sess, s_size, a_size, self.name, None)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        epsilon = 0.1
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                epsilon = epsilon * 0.995

                while not d:
                    if not ROLLING_EVENT.is_set():
                        ROLLING_EVENT.wait()
                    GLOBAL_STEP += 1

                    if random.random() > epsilon:
                        a_dist_list = sess.run(self.local_Apex.Q, feed_dict={self.local_Apex.inputs:[s]})
                        a_dist = a_dist_list[0]
                        a = np.argmax(a_dist)
                    else:
                        a = random.randint(0, 5)
                        
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    self.lock.acquire()
                    try:
                        replaymemory.add([s,a,r,s1,d])
                    finally:
                        self.lock.release()
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    
                    if d != True:
                        sess.run(self.update_local_ops)
                    else:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                    
                    #if episode_count % 50 == 0 and self.name == 'worker_0':
                    #    saver.save(sess,self.model_path+'/last-'+str(episode_count)+'.cptk')
                    #    print ("Saved Model")
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward
                        
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99
s_size = 7056
load_model = False
model_path = './last'
N = 20
k = 1.
benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.00015)
num_workers = 4
batch_size = 10
max_memory = 300000
replaymemory = ReplayMemory(max_memory)
saver = tf.train.Saver(max_to_keep=5)
lock = threading.Lock()

with tf.Session() as sess:
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    
    GLOBAL_STEP = 0
    coord = tf.train.Coordinator()
    master_network = Apex_Network(sess, s_size,a_size,'global',trainer)
    workers = []
    for i in range(num_workers):
        env = get_env(task)
        workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes,lock))
    
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    worker_threads.append(threading.Thread(target=master_network.train(sess, gamma)))
    
    worker_threads[-1].start()
    coord.join(worker_threads)

