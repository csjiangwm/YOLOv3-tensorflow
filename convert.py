# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 08:56:41 2018

@author: jwm
"""
import os
import configparser
import io
import numpy as np
from collections import defaultdict
from collections import namedtuple
import tensorflow as tf
from models.yolov3 import yolov3
from utils.utils import process_config


def unique_config_sections(config_file):
    """
    Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.BytesIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

# %%
def load_weights_for_finetune(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    output_path = os.path.expanduser(args.npz_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.74'), '{} is not a needed weights file'.format(weights_path)
    assert output_path.endswith('.npz'), 'output path {} is not a .npz file'.format(output_path)
    
    conv_bn = namedtuple('conv_bn', ['bias', 'gamma', 'mean', 'variance', 'conv_weights'])
    weights_dict = {}
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    print('Weights Header: ', major, minor, revision, seen)
    
    filter_list = [3, ]
    
    for section in cfg_parser.sections():
        if len(filter_list) > 74:
            break
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
#            stride = int(cfg_parser[section]['stride'])
#            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section] # True or False
#            padding = 'same' if pad == 1 else 'valid'                         # padding='same' is equivalent to Darknet pad=1
            # Setting weights: darknet serializes convolutional weights as: [bias/beta, [gamma, mean, variance], conv_weights]
            weights_shape = (size, size, filter_list[-1], filters) # TODO: This assumes channel last dim_ordering.
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)
            
            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)
    
            conv_bias = np.ndarray(shape=(filters,),dtype='float32',buffer=weights_file.read(filters * 4))
            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters),dtype='float32',buffer=weights_file.read(filters * 3 * 4))
            conv_weights = np.ndarray(shape=darknet_w_shape,dtype='float32',buffer=weights_file.read(weights_size * 4))
            # DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order: (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0]) # TODO: Add check for Theano dim ordering.
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]
            filter_list.append(filters)
            weights = conv_bn(bias=conv_bias, gamma=bn_weights[0], mean=bn_weights[1], variance=bn_weights[2],conv_weights=conv_weights)
            section = 'conv_' + section.split('_')[-1]
            weights_dict[section] = weights
        elif section.startswith('shortcut'):
            from_list = [int(l) for l in (cfg_parser[section]['from']).strip().split(',')]
            assert from_list[0] < 0, 'relative coord'
            c_ = filter_list[from_list[0]]
            print('shortcut #channel:{}'.format(c_))
            filter_list.append(c_)
        elif section.startswith('net'):
            pass
        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))
                
    weights_file.close()
    np.savez(output_path, **weights_dict)
    
def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5) # read 5*4=20 bytes at first
        weights = np.fromfile(fp, dtype=np.float32) # read the rests

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        print "%d-th iteration..." % i
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, bias, mean, variance = var_list[i + 1:i + 5]
                batch_norm_vars = [bias, gamma, mean, variance]
                for _var in batch_norm_vars:
                    print _var
                    shape = _var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(_var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv' in var2.name.split('/')[-2]:
                print var2
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            print var1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    return assign_ops
    
    
    
if __name__ == '__main__':
    try:
        FLAG = process_config()
    except:
        print("missing or invalid arguments")
        exit(0)
        
    if FLAG.npz:
        if os.path.exists(FLAG.npz_path):
            print "darknet53.conv.74.npz already exists"
        else:
            print FLAG.config_path
            load_weights_for_finetune(FLAG)
            
    elif FLAG.ckpt:
        detections = yolov3(FLAG)
#        with tf.variable_scope('detector'):
        detections.build()
        load_ops = load_weights(tf.global_variables(), FLAG.weights_path)
        detections.init_saver()
        with tf.Session() as sess:
            sess.run(load_ops)
            writer =tf.summary.FileWriter("experiments/summary/",graph = sess.graph)
            writer.close()
            detections.saver.save(sess,"experiments/ckpt/yolov3.ckpt")
    else:
        raise ValueError('Missing important parameters')