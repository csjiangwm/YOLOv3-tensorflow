# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""
import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.ckpt_dir,'train.model'), self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.ckpt_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        else:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
        
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch',reuse=tf.AUTO_REUSE):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step',reuse=tf.AUTO_REUSE):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError
