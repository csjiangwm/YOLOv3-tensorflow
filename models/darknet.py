# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:55:27 2018

@author: jwm
"""

import tensorflow as tf
import numpy as np
conv_bn = ['bias', 'gamma', 'mean', 'variance', 'conv_weights']

class Darknet53:
    """
    Builds Darknet-53 model.
    """
    def __init__(self, config):
        """
        :param trainable:   python
        """
        self.config = config
        if self.config.is_restarting and not self.config.ckpt:
            self.data_dict = np.load(self.config.npz_path)
            self.keys = self.data_dict.keys()

    def get_var(self, initial_value, name, var_name):
        """

        :param initial_value:
        :param name:
        :param var_name:
        :param trainable:  moving average not trainable
        :return:
        """
        if not self.config.is_restarting or self.config.ckpt:
            value = initial_value
        elif self.data_dict is not None and name in self.keys:
            idx = conv_bn.index(var_name)
            value = self.data_dict[name][idx] if idx < 4 else self.data_dict[name][idx][0]
        else:
            raise ValueError('From scratch train feature extractor or provide complete weights')

        if self.config.is_training:
            var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.const(value, dtype=tf.float32, name=var_name)
        return var

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 'conv_weights')
        return filters
    
    def fixed_padding(self, inputs, kernel_size, mode='CONSTANT', **kwargs):
        """
        Pads the input along the spatial dimensions independently of input size.
    
        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                       Should be a positive integer.
          data_format: The input format ('NHWC' or 'NCHW').
          mode: The mode for tf.pad.
    
        Returns:
          A tensor with the same format as the input with the data either intact
          (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
    
        if kwargs['data_format'] == 'NCHW':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],[pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],[pad_beg, pad_end], [0, 0]], mode=mode)
        return padded_inputs
    
    def conv_layer(self, inputs, size, stride, in_channels, out_channels, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            if stride > 1:
#                inputs = self.fixed_padding(inputs,size,data_format='NHWC')
                inputs = tf.pad(inputs, [[0, 0], [1, 0],[1, 0], [0, 0]], mode='CONSTANT')# Darknet uses left and top padding instead of 'same' mode
            filt = self.get_conv_var(size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(inputs, filt, [1, stride, stride, 1], padding=('SAME' if stride == 1 else 'VALID'))
#            normed = tf.layers.batch_normalization(conv, training=self.phase_train,name='BatchNorm')
            normed = tf.contrib.slim.batch_norm(conv,decay=self.decay_bn,epsilon=1e-5,scale=True,is_training=self.phase_train,fused=None)
            activation = tf.nn.leaky_relu(normed, alpha=0.1)
        return activation

    def build(self, inputs, istraining, decay_bn=0.9):
        self.phase_train = tf.constant(istraining,tf.bool)
        self.decay_bn = decay_bn
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv0 = self.conv_layer(inputs=inputs, size=3, stride=1, in_channels=3, out_channels=32, name='conv_0')             # 416x416x32
# %% padding, convolution, batch_normalization, and leaky_relu activation
        self.conv1 = self.conv_layer(inputs=self.conv0, size=3, stride=2, in_channels=32, out_channels=64, name='conv_1')        # 208x208x64
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv2 = self.conv_layer(inputs=self.conv1, size=1, stride=1, in_channels=64, out_channels=32, name='conv_2')        # 208x208x32
        self.conv3 = self.conv_layer(inputs=self.conv2, size=3, stride=1, in_channels=32, out_channels=64, name='conv_3')        # 208x208x64
        self.res0 = self.conv3 + self.conv1                                         # 208x208x64
# %% padding, convolution, batch_normalization, and leaky_relu activation
        self.conv4 = self.conv_layer(inputs=self.res0, size=3, stride=2, in_channels=64, out_channels=128, name='conv_4')        # 104x104x128
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv5 = self.conv_layer(inputs=self.conv4, size=1, stride=1, in_channels=128, out_channels=64, name='conv_5')       # 104x104x64
        self.conv6 = self.conv_layer(inputs=self.conv5, size=3, stride=1, in_channels=64, out_channels=128, name='conv_6')       # 104x104x128
        self.res1 = self.conv6 + self.conv4     # 128                               # 104x104x128
        self.conv7 = self.conv_layer(inputs=self.res1, size=1, stride=1, in_channels=128, out_channels=64, name='conv_7')        # 104x104x64
        self.conv8 = self.conv_layer(inputs=self.conv7, size=3, stride=1, in_channels=64, out_channels=128, name='conv_8')       # 104x104x128
        self.res2 = self.conv8 + self.res1      # 128                               # 104x104x128
# %% padding, convolution, batch_normalization, and leaky_relu activation
        self.conv9 = self.conv_layer(inputs=self.res2, size=3, stride=2, in_channels=128, out_channels=256, name='conv_9')       # 52x52x256
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv10 = self.conv_layer(inputs=self.conv9, size=1, stride=1, in_channels=256, out_channels=128, name='conv_10')    # 52x52x128
        self.conv11 = self.conv_layer(inputs=self.conv10, size=3, stride=1, in_channels=128, out_channels=256, name='conv_11')   # 52x52x256
        self.res3 = self.conv11 + self.conv9                                        # 52x52x256
        self.conv12 = self.conv_layer(inputs=self.res3, size=1, stride=1, in_channels=256, out_channels=128, name='conv_12')     # 52x52x128
        self.conv13 = self.conv_layer(inputs=self.conv12, size=3, stride=1, in_channels=128, out_channels=256, name='conv_13')   # 52x52x256
        self.res4 = self.conv13 + self.res3                                         # 52x52x256
        self.conv14 = self.conv_layer(inputs=self.res4, size=1, stride=1, in_channels=256, out_channels=128, name='conv_14')     # 52x52x128
        self.conv15 = self.conv_layer(inputs=self.conv14, size=3, stride=1, in_channels=128, out_channels=256, name='conv_15')   # 52x52x256
        self.res5 = self.conv15 + self.res4                                         # 52x52x256
        self.conv16 = self.conv_layer(inputs=self.res5, size=1, stride=1, in_channels=256, out_channels=128, name='conv_16')     # 52x52x128
        self.conv17 = self.conv_layer(inputs=self.conv16, size=3, stride=1, in_channels=128, out_channels=256, name='conv_17')   # 52x52x256
        self.res6 = self.conv17 + self.res5                                         # 52x52x256
        self.conv18 = self.conv_layer(inputs=self.res6, size=1, stride=1, in_channels=256, out_channels=128, name='conv_18')     # 52x52x128
        self.conv19 = self.conv_layer(inputs=self.conv18, size=3, stride=1, in_channels=128, out_channels=256, name='conv_19')   # 52x52x256
        self.res7 = self.conv19 + self.res6                                         # 52x52x256
        self.conv20 = self.conv_layer(inputs=self.res7, size=1, stride=1, in_channels=256, out_channels=128, name='conv_20')     # 52x52x128
        self.conv21 = self.conv_layer(inputs=self.conv20, size=3, stride=1, in_channels=128, out_channels=256, name='conv_21')   # 52x52x256
        self.res8 = self.conv21 + self.res7                                         # 52x52x256
        self.conv22 = self.conv_layer(inputs=self.res8, size=1, stride=1, in_channels=256, out_channels=128, name='conv_22')     # 52x52x128
        self.conv23 = self.conv_layer(inputs=self.conv22, size=3, stride=1, in_channels=128, out_channels=256, name='conv_23')   # 52x52x256
        self.res9 = self.conv23 + self.res8                                         # 52x52x256
        self.conv24 = self.conv_layer(inputs=self.res9, size=1, stride=1, in_channels=256, out_channels=128, name='conv_24')     # 52x52x128
        self.conv25 = self.conv_layer(inputs=self.conv24, size=3, stride=1, in_channels=128, out_channels=256, name='conv_25')   # 52x52x256
        self.res10 = self.conv25 + self.res9                                        # 52x52x256
# %% padding, convolution, batch_normalization, and leaky_relu activation
        self.conv26 = self.conv_layer(inputs=self.res10, size=3, stride=2, in_channels=256, out_channels=512, name='conv_26')    # 26x26x512
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv27 = self.conv_layer(inputs=self.conv26, size=1, stride=1, in_channels=512, out_channels=256, name='conv_27')   # 26x26x256
        self.conv28 = self.conv_layer(inputs=self.conv27, size=3, stride=1, in_channels=256, out_channels=512, name='conv_28')   # 26x26x512
        self.res11 = self.conv28 + self.conv26                                      # 26x26x512
        self.conv29 = self.conv_layer(inputs=self.res11, size=1, stride=1, in_channels=512, out_channels=256, name='conv_29')    # 26x26x256
        self.conv30 = self.conv_layer(inputs=self.conv29, size=3, stride=1, in_channels=256, out_channels=512, name='conv_30')   # 26x26x512
        self.res12 = self.conv30 + self.res11                                       # 26x26x512
        self.conv31 = self.conv_layer(inputs=self.res12, size=1, stride=1, in_channels=512, out_channels=256, name='conv_31')    # 26x26x256
        self.conv32 = self.conv_layer(inputs=self.conv31, size=3, stride=1, in_channels=256, out_channels=512, name='conv_32')   # 26x26x512
        self.res13 = self.conv32 + self.res12                                       # 26x26x512
        self.conv33 = self.conv_layer(inputs=self.res13, size=1, stride=1, in_channels=512, out_channels=256, name='conv_33')    # 26x26x256
        self.conv34 = self.conv_layer(inputs=self.conv33, size=3, stride=1, in_channels=256, out_channels=512, name='conv_34')   # 26x26x512
        self.res14 = self.conv34 + self.res13                                       # 26x26x512
        self.conv35 = self.conv_layer(inputs=self.res14, size=1, stride=1, in_channels=512, out_channels=256, name='conv_35')    # 26x26x256
        self.conv36 = self.conv_layer(inputs=self.conv35, size=3, stride=1, in_channels=256, out_channels=512, name='conv_36')   # 26x26x512
        self.res15 = self.conv36 + self.res14                                       # 26x26x512
        self.conv37 = self.conv_layer(inputs=self.res15, size=1, stride=1, in_channels=512, out_channels=256, name='conv_37')    # 26x26x256
        self.conv38 = self.conv_layer(inputs=self.conv37, size=3, stride=1, in_channels=256, out_channels=512, name='conv_38')   # 26x26x512
        self.res16 = self.conv38 + self.res15                                       # 26x26x512
        self.conv39 = self.conv_layer(inputs=self.res16, size=1, stride=1, in_channels=512, out_channels=256, name='conv_39')    # 26x26x256
        self.conv40 = self.conv_layer(inputs=self.conv39, size=3, stride=1, in_channels=256, out_channels=512, name='conv_40')   # 26x26x512
        self.res17 = self.conv40 + self.res16                                       # 26x26x512
        self.conv41 = self.conv_layer(inputs=self.res17, size=1, stride=1, in_channels=512, out_channels=256, name='conv_41')    # 26x26x256
        self.conv42 = self.conv_layer(inputs=self.conv41, size=3, stride=1, in_channels=256, out_channels=512, name='conv_42')   # 26x26x512
        self.res18 = self.conv42 + self.res17                                       # 26x26x512
# %% padding, convolution, batch_normalization, and leaky_relu activation
        self.conv43 = self.conv_layer(inputs=self.res18, size=3, stride=2, in_channels=512, out_channels=1024, name='conv_43')   # 13x13x1024
# %%          convolution, batch_normalization, and leaky relu activation
        self.conv44 = self.conv_layer(inputs=self.conv43, size=1, stride=1, in_channels=1024, out_channels=512, name='conv_44')  # 13x13x512
        self.conv45 = self.conv_layer(inputs=self.conv44, size=3, stride=1, in_channels=512, out_channels=1024, name='conv_45')  # 13x13x1024
        self.res19 = self.conv45 + self.conv43                                      # 13x13x1024
        self.conv46 = self.conv_layer(inputs=self.res19, size=1, stride=1, in_channels=1024, out_channels=512, name='conv_46')   # 13x13x512
        self.conv47 = self.conv_layer(inputs=self.conv44, size=3, stride=1, in_channels=512, out_channels=1024, name='conv_47')  # 13x13x1024
        self.res20 = self.conv47 + self.res19                                       # 13x13x1024
        self.conv48 = self.conv_layer(inputs=self.res20, size=1, stride=1, in_channels=1024, out_channels=512, name='conv_48')   # 13x13x512
        self.conv49 = self.conv_layer(inputs=self.conv48, size=3, stride=1, in_channels=512, out_channels=1024, name='conv_49')  # 13x13x1024
        self.res21 = self.conv49 + self.res20                                       # 13x13x1024
        self.conv50 = self.conv_layer(inputs=self.res21, size=1, stride=1, in_channels=1024, out_channels=512, name='conv_50')   # 13x13x512
        self.conv51 = self.conv_layer(inputs=self.conv50, size=3, stride=1, in_channels=512, out_channels=1024, name='conv_51')  # 13x13x1024
        self.res22 = self.conv51 + self.res21                                       # 13x13x1024
        
        return self.res22, self.res18, self.res10
             # 13x13x1024   26x26x512  52x52x256  
        