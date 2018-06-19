# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:03:40 2018

@author: jwm
"""

import tensorflow as tf


class yolo_scale:
    """
        Multi-scale
    """
    def __init__(self, config):
        self.config = config

    def conv_layer(self, inputs, size, stride, in_channels, out_channels, use_bn, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(inputs, out_channels, size, stride, padding="SAME", use_bias=not use_bn, activation=None)
            if use_bn:
                conv_bn = tf.layers.batch_normalization(conv, training=self.config.is_training,name='BatchNorm')
#                conv_bn = tf.contrib.slim.batch_norm(conv,decay=0.9,epsilon=1e-5,scale=True,is_training=self.config.is_training,fused=None)
                act = tf.nn.leaky_relu(conv_bn, 0.1)
            else:
                act = conv
        return act
        
    def upsample(self, inputs, out_shape):
        padded_inputs = tf.pad(inputs, [[0, 0], [1, 1],[1, 1], [0, 0]], mode='SYMMETRIC')
        height = out_shape[2]
        width = out_shape[1]
        # we padded with 1 pixel from each side and upsample by factor of 2, so new dimensions will be greater by 4 pixels after interpolation
        new_height = height + 4
        new_width = width + 4
        outputs = tf.image.resize_bilinear(padded_inputs, (new_height, new_width))
        # trim back to desired size
        outputs = outputs[:, 2:-2, 2:-2, :]
        outputs = tf.identity(outputs, name='upsampled')
        return outputs
        
    def build(self, feat_ex, res18, res10): # 13x13x1024   26x26x512  52x52x256  
        # %%
        self.conv52 = self.conv_layer(feat_ex, 1, 1, 1024, 512, True, 'conv_head_52')       # 13x13x512
        self.conv53 = self.conv_layer(self.conv52, 3, 1, 512, 1024, True, 'conv_head_53')   # 13x13x1024
        self.conv54 = self.conv_layer(self.conv53, 1, 1, 1024, 512, True, 'conv_head_54')   # 13x13x512
        self.conv55 = self.conv_layer(self.conv54, 3, 1, 512, 1024, True, 'conv_head_55')   # 13x13x1024
        self.conv56 = self.conv_layer(self.conv55, 1, 1, 1024, 512, True, 'conv_head_56')   # 13x13x512
        
        self.conv57 = self.conv_layer(self.conv56, 3, 1, 512, 1024, True, 'conv_head_57')   # 13x13x1024
        self.conv58 = self.conv_layer(self.conv57, 1, 1, 1024, 3*(5+self.config.classes), False, 'conv_head_58')         # 13x13x255 !!!
        # %% follow yolo layer mask = 6,7,8   scale:1
        self.conv59 = self.conv_layer(self.conv56, 1, 1, 512, 256, True, 'conv_head_59')    # 13x13x256
        upsample_size = res18.get_shape().as_list()
#        self.upsample0 = tf.image.resize_bilinear(self.conv59, [2*upsample_size, 2*upsample_size],name='upsample_0')    # 26x26x256
        upsample0 = self.upsample(self.conv59,upsample_size)
        self.route0 = tf.concat([upsample0, res18], axis=3, name='route_0')           # 26x26x768
        self.conv60 = self.conv_layer(self.route0, 1, 1, 768, 256, True, 'conv_head_60')    # 26x26x256
        self.conv61 = self.conv_layer(self.conv60, 3, 1, 256, 512, True, 'conv_head_61')    # 26x26x512
        self.conv62 = self.conv_layer(self.conv61, 1, 1, 512, 256, True, 'conv_head_62')    # 26x26x256
        self.conv63 = self.conv_layer(self.conv62, 3, 1, 256, 512, True, 'conv_head_63')    # 26x26x512
        self.conv64 = self.conv_layer(self.conv63, 1, 1, 512, 256, True, 'conv_head_64')    # 26x26x256
        
        self.conv65 = self.conv_layer(self.conv64, 3, 1, 256, 512, True, 'conv_head_65')    # 26x26x512
        self.conv66 = self.conv_layer(self.conv65, 1, 1, 512, 3*(5+self.config.classes), False, 'conv_head_66')           # 26x26x255 !!!
        # %% follow yolo layer mask = 3,4,5  scale:2
        self.conv67 = self.conv_layer(self.conv64, 1, 1, 256, 128, True, 'conv_head_67')    # 26x26x128
        upsample_size = res10.get_shape().as_list()
#        self.upsample1 = tf.image.resize_bilinear(self.conv67, [2 * upsample_size, 2 * upsample_size],name='upsample_1') # 52x52x128
        upsample1 = self.upsample(self.conv67,upsample_size)
        self.route1 = tf.concat([upsample1, res10], axis=3, name='route_1')           # 52x52x384
        self.conv68 = self.conv_layer(self.route1, 1, 1, 384, 128, True, 'conv_head_68')    # 52x52x128
        self.conv69 = self.conv_layer(self.conv68, 3, 1, 128, 256, True, 'conv_head_69')    # 52x52x256
        self.conv70 = self.conv_layer(self.conv69, 1, 1, 256, 128, True, 'conv_head_70')    # 52x52x128
        self.conv71 = self.conv_layer(self.conv70, 3, 1, 128, 256, True, 'conv_head_71')    # 52x52x256
        self.conv72 = self.conv_layer(self.conv71, 1, 1, 256, 128, True, 'conv_head_72')    # 52x52x128
        
        self.conv73 = self.conv_layer(self.conv72, 3, 1, 128, 256, True, 'conv_head_73')    # 52x52x256
        self.conv74 = self.conv_layer(self.conv73, 1, 1, 256, 3*(5+self.config.classes), False, 'conv_head_74')           # 52x52x255 !!!
        # %% follow yolo layer mask = 0,1,2  scale:3
        return self.conv74, self.conv66, self.conv58


class yolo_head:
    """
        Convert final layer features to bounding box parameters.

        Parameters
        ----------
        feats : tensor
            Final convolutional layer features.
        anchors : array-like
            Anchor box widths and heights.
        num_classes : int
            Number of target classes.

        Returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution.
        box_conf : tensor
            Probability estimate for whether each box contains any object.
        box_class_pred : tensor
            Probability distribution estimate for each box over class labels.
    """
    def __init__(self, anchors, num_classes, img_shape, num_anchors_per_layer):
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.num_anchors_per_layer = num_anchors_per_layer

    def build(self, feats):
        # Reshapce to bach, height, widht, num_anchors, box_params
        anchors_tensor = tf.reshape(self.anchors, [1, 1, 1, self.num_anchors_per_layer, 2]) # [1,1,1,3,2]

        # Dynamic implementation of conv dims for fully convolutional model
        conv_dims = tf.stack([tf.shape(feats)[2], tf.shape(feats)[1]])# height, width ---> 13x13 for scale1; 
        #                                                                                  26x26 for scale2; 
        #                                                                                  52x52 for scale3
        # In YOLO the height index is the inner most iteration
        conv_height_index = tf.range(conv_dims[1]) # range(13/26/52)
        conv_width_index = tf.range(conv_dims[0])  # range(13/26/52)
        conv_width_index, conv_height_index = tf.meshgrid(conv_width_index, conv_height_index)
        conv_height_index = tf.reshape(conv_height_index, [-1, 1])
        conv_width_index = tf.reshape(conv_width_index, [-1, 1])
        conv_index = tf.concat([conv_width_index, conv_height_index], axis=-1)
        # 0, 0
        # 1, 0
        # 2, 0
        # ...
        # 12, 0
        # 0, 1
        # 1, 1
        # ...
        # 12, 1
        conv_index = tf.reshape(conv_index, [1, conv_dims[1], conv_dims[0], 1, 2])  # [1, 13, 13, 1, 2]
        conv_index = tf.cast(conv_index, tf.float32)

        feats = tf.reshape(feats, [-1, conv_dims[1], conv_dims[0], self.num_anchors_per_layer, self.num_classes + 5]) # [None, 13, 13, 3, 85]

        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32)

        img_dims = tf.stack([self.img_shape[2], self.img_shape[1]])   # w, h
        img_dims = tf.cast(tf.reshape(img_dims, [1, 1, 1, 1, 2]), tf.float32)

        box_xy = tf.sigmoid(feats[..., :2])  # σ(tx), σ(ty)     # [None, 13, 13, 3, 2]
        
        box_twh = feats[..., 2:4]
        box_wh = tf.exp(box_twh)  # exp(tw), exp(th)    # [None, 13, 13, 3, 2]
        self.box_confidence = tf.sigmoid(feats[..., 4:5])
        self.box_class_probs = tf.sigmoid(feats[..., 5:])        # multi-label classification

        self.box_xy = (box_xy + conv_index) / conv_dims  # relative the whole img [0, 1]
        self.box_wh = box_wh * anchors_tensor / img_dims  # relative the whole img [0, 1]
        self.loc_txywh = tf.concat([box_xy, box_twh], axis=-1)

        return self.box_xy, self.box_wh, self.box_confidence, self.box_class_probs, self.loc_txywh
        # box_xy: [None, 13, 13, 3, 2]
        # box_wh: [None, 13, 13, 3, 2]
        # box_confidence: [None, 13, 13, 3, 1]
        # box_class_probs: [None, 13, 13, 3, 80]
        # loc_txywh: [None, 13, 13, 3, 2+2]
    
    
    