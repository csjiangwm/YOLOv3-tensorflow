# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import argparse
import os
from configs.config import cfg as FLAG
import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_args():
    """
    read and return argument
    """
    parser = argparse.ArgumentParser(description='parameter')
    
    parser.add_argument('--weights_path', default='../darknet/darknet53.conv.74', help='Path to Darknet weights file.')
    parser.add_argument('--config_path', default='configs/yolov3.cfg',  help='Path to Darknet cfg file.')
    parser.add_argument('--npz_path', default='experiments/darknet53.conv.74.npz', help='Path to output npz model file.')
    parser.add_argument('--xml_path', default='../VOC2007/Annotations', help='Path to xml file.')
    parser.add_argument('--imgs_path', default='../VOC2007/JPEGImages', help='Path to image file.')
    parser.add_argument('--summary_dir', default='experiments/summary', help='Path to summary directory.')
    parser.add_argument('--ckpt_dir', default='experiments/ckpt', help='Path to checkpoint directory.')
    parser.add_argument('--GPU_options', default=1, help='Path to Darknet weights file.', type=int)
    parser.add_argument('--classes_path', default='./configs/coco_classes.txt', help='Path to class name file.')
    parser.add_argument('--train', default=True, help='train flag.', type=bool)
    # if True, restart training with darknet53.conv.74.npz, else restart training with last checkpoint
    parser.add_argument('--restart', default=True, help='restart training.', type=bool)
    # for convert.py 
    parser.add_argument('--npz', default=False, help='Transform to .npz.', type=bool)
    parser.add_argument('--ckpt', default=False, help='Tranform to .ckpt.', type=bool)
    
    args = parser.parse_args()
    return args
    
def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
        
def read_classes(classes_path):
    """
    return class names
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
def process_config():
    """
    package arguments and configures
    """

    args = get_args()
    FLAG.names = read_classes(args.classes_path)
    FLAG.classes = len(FLAG.names)
    
    FLAG.GPU_options = args.GPU_options
    FLAG.is_restarting = args.restart
    FLAG.is_training = args.train
    FLAG.weights_path = args.weights_path
    FLAG.config_path = args.config_path
    FLAG.npz_path = args.npz_path
    FLAG.xml_path = args.xml_path
    FLAG.imgs_path = args.imgs_path
    FLAG.summary_dir = args.summary_dir
    FLAG.ckpt_dir = args.ckpt_dir
    FLAG.npz = args.npz
    FLAG.ckpt = args.ckpt

    create_dirs([FLAG.summary_dir, FLAG.ckpt_dir])
    return FLAG


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return np.array(image)


