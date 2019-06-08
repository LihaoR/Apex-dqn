#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:45:19 2018

@author: lihaoruo
"""
from actor_mp import Actor
from learner_mp import Learner

import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
import gym
from atari_wrappers import wrap_deepmind
from time import sleep

# Used to set worker network parameters to those of global network.
def actor_work(i,queues,env):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth=True 
    sess = tf.Session(config=config)
    
    actor = Actor(sess,env,i,s_size,a_size,queues)
    actor.run()

def learner_work(scope,queues,env):
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth=True 
    sess = tf.Session(config=config)
    
    trainer = tf.train.AdamOptimizer(learning_rate=0.00015)
    leaner = Learner(sess,s_size,a_size,scope,queues,trainer)
    leaner.run(gamma, s_size, a_size, batch_size=128,env=env)


def get_env(task):
    env = gym.make(task)
    env = wrap_deepmind(env)
    return env

gamma = .99
s_size = 7056
load_model = False
model_path = './model'
N = 20
k = 1.
task = 'PongNoFrameskip-v4'
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

num_actors = 2
#saver = tf.train.Saver(max_to_keep=5)

if __name__ == '__main__':
    transition_queue = mp.Queue(400)
    param_queue = mp.Queue(num_actors)
    ps = []
    #ps.append(mp.Process(target=learner_work,args=('learner',(transition_queue,param_queue))))
    for i in range(num_actors):
        ps.append(mp.Process(target=actor_work,args=(str(i),(transition_queue,param_queue),env)))
    
    ps.append(mp.Process(target=learner_work,args=('learner',(transition_queue,param_queue),env)))
    for p in ps:
        p.start()
        sleep(0.2)

    for p in ps:
        p.join()


