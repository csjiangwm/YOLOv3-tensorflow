# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.yolov3 import yolov3
from trainers.yolov3_trainer import yolov3_trainer
from utils.utils import process_config
from utils.logger import Logger


def main():
    try:
        FLAG = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
    
    if FLAG.GPU_options:
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session_config.gpu_options.allow_growth = True
        sess = tf.Session(config=session_config)
    else:
        sess = tf.Session()
        
    
    model = yolov3(FLAG)
    data = DataGenerator(FLAG)
    
    logger = Logger(sess, FLAG)
    trainer = yolov3_trainer(sess, model, data, FLAG, logger)
    print "Start training..."
    trainer.train()
#        
    sess.close()


if __name__ == '__main__':
    main()
