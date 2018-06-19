# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class yolov3_trainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(yolov3_trainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.train.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        return loss

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch())
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss = self.sess.run([self.model.train_op, self.model.loss],feed_dict=feed_dict)
        return loss
