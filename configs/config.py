# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:45:04 2018

@author: jwm
"""

from easydict import EasyDict as edict
import numpy as np


__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.trained_img_num = 5011


__C.anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])

__C.num = 9
__C.num_anchors_per_layer = 3
__C.batch_size = 4

#
# Training options
#
__C.train = edict()

__C.train.ignore_thresh = .5
__C.train.momentum = 0.9
__C.train.decay = 0.0005
__C.train.learning_rate = 0.001
__C.train.max_batches = 502000
__C.train.lr_steps = [40000, 45000]
__C.train.lr_scales = [.1, .1]
__C.train.max_truth = 50
__C.train.mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
__C.train.image_resized = 416   # { 320, 352, ... , 608} multiples of 32
__C.train.image_ranged = [320,416,352,416,384,416,448,416,480,416,512,416,544,416,576,416,608,416]
__C.train.num_iter_per_epoch = 1000
__C.train.num_epochs = 2000
__C.train.random = 0
