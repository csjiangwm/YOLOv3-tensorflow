# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""


class BaseTrain(object):
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        
        self.model.init_global_step()
        self.model.init_cur_epoch()
        
        
    def build_training(self):
        self.model.build()
        self.model.compute_loss()
        self.model.optimize()
        self.model.load(self.sess)
        

    def train(self):
        '''
        start training
        '''
        self.model.init_saver()
        self.build_training()
        
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.train.num_epochs + 1, 1):
            loss = self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            print "The %d-th iteration's loss is: %f" % (cur_epoch, loss)
            if cur_epoch > 5 and loss < 0.001:
                break

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
