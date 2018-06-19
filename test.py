# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:41:04 2018

@author: jwm
"""

import tensorflow as tf

from models.yolov3 import yolov3
import numpy as np
from PIL import Image
from utils.utils import process_config,draw_boxes
import matplotlib.pyplot as plt
import time


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
    model.build()
    model.init_saver()
    model.load(sess)

    image_test = Image.open('images/timg.jpg')
    resized_image = image_test.resize(size=(416, 416))
    image_data = np.array(resized_image, dtype='float32')/255.0
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.5)

    begin_time = time.time()
    boxes_, scores_, classes_, conv0 = sess.run([boxes, scores, classes, model.feature_extractor.conv0],feed_dict={img_hw: [image_test.size[1], image_test.size[0]],
                                                                             model.x: [image_data]})
    end_time = time.time()
    print (end_time - begin_time)
#    print conv0
    
    image_draw = draw_boxes(np.array(image_test, dtype=np.float32) / 255, boxes_, classes_, FLAG.names, scores=scores_)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image_draw)
    fig.savefig('prediction.jpg')
    plt.show()
    sess.close()


if __name__ == '__main__':
    main()
