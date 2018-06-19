# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

from base.base_model import BaseModel
import tensorflow as tf
from models.darknet import Darknet53
from models.yolo_layer import yolo_scale,yolo_head


class yolov3(BaseModel):
    '''
        The whole YOLOv3 model contains darknet, yolo body, yolo head and yolo loss
    '''
    def __init__(self, config, decay_bn=0.9):
        super(yolov3, self).__init__(config)
        self.config = config
        self.istraining = self.config.is_training
        self.decay_bn = decay_bn
        
    def init_saver(self):
        self.saver = tf.train.Saver()

    def build(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.train.image_resized,self.config.train.image_resized,3])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.train.max_truth, 5])
        self.img_shape = tf.shape(self.x)
        
        with tf.variable_scope("Feature_Extractor",reuse=tf.AUTO_REUSE):
            self.feature_extractor = Darknet53(self.config)
            self.feats52,self.route2,self.route1 = self.feature_extractor.build(self.x, self.istraining, self.decay_bn) # ?x13x13x1024
#           self.feats52   ==>  darknet conv52  --> Nonex13x13x1024
#           self.route2    ==>  darknet res18   --> Nonex26x26x512   
#           self.route1    ==>  darknet res10   --> Nonex52x52x256
            
        with tf.variable_scope("Scale",reuse=tf.AUTO_REUSE):
            scale = yolo_scale(self.config)
            self.yolo123, self.yolo456, self.yolo789 = scale.build(self.feats52,self.route2,self.route1)
#           self.yolo123   ==>  mask: 0,1,2   --> Nonex52x52x255
#           self.yolo456   ==>  make: 3,4,5   --> Nonex26x26x255
#           self.yolo789   ==>  make: 7.8.9   --> Nonex13x13x255
    
        with tf.variable_scope("Detection_0",reuse=tf.AUTO_REUSE):
            self.anchors0 = tf.constant(self.config.anchors[self.config.train.mask[0]], dtype=tf.float32) #[10, 13], [16, 30], [33, 23]
            head = yolo_head(self.anchors0, self.config.classes, self.img_shape, self.config.num_anchors_per_layer)
            self.pred_xy0, self.pred_wh0, self.pred_confidence0, self.pred_class_prob0, self.loc_txywh0 = head.build(self.yolo123)
            
        with tf.variable_scope("Detection_1",reuse=tf.AUTO_REUSE):
            self.anchors1 = tf.constant(self.config.anchors[self.config.train.mask[1]], dtype=tf.float32) #[30, 61], [62, 45], [59, 119]
            head = yolo_head(self.anchors1, self.config.classes, self.img_shape, self.config.num_anchors_per_layer)
            self.pred_xy1, self.pred_wh1, self.pred_confidence1, self.pred_class_prob1, self.loc_txywh1 = head.build(self.yolo456)
            
        with tf.variable_scope("Detection_2",reuse=tf.AUTO_REUSE):
            self.anchors2 = tf.constant(self.config.anchors[self.config.train.mask[2]], dtype=tf.float32) #[116, 90], [156, 198], [373, 326]
            head = yolo_head(self.anchors2, self.config.classes, self.img_shape, self.config.num_anchors_per_layer)
            self.pred_xy2, self.pred_wh2, self.pred_confidence2, self.pred_class_prob2, self.loc_txywh2 = head.build(self.yolo789)
            
    def compute_loss(self):
        with tf.name_scope('Loss_0'):
            matching_true_boxes, detectors_mask, loc_scale = self.preprocess_true_boxes(self.y,
                                                                                   self.anchors0,
                                                                                   tf.shape(self.yolo123),
                                                                                   self.img_shape)
            objectness_loss = self.confidence_loss(self.pred_xy0, self.pred_wh0, self.pred_confidence0, self.y, detectors_mask)
            cord_loss = self.cord_cls_loss(detectors_mask, matching_true_boxes,
                                      self.config.classes, self.pred_class_prob0, self.loc_txywh0, loc_scale)
            loss1 = objectness_loss + cord_loss
        with tf.name_scope('Loss_1'):
            matching_true_boxes, detectors_mask, loc_scale = self.preprocess_true_boxes(self.y,
                                                                                   self.anchors1,
                                                                                   tf.shape(self.yolo456),
                                                                                   self.img_shape)
            objectness_loss = self.confidence_loss(self.pred_xy1, self.pred_wh1, self.pred_confidence1, self.y, detectors_mask)
            cord_loss = self.cord_cls_loss(detectors_mask, matching_true_boxes,
                                      self.config.classes, self.pred_class_prob1, self.loc_txywh1, loc_scale)
            loss2 = objectness_loss + cord_loss
        with tf.name_scope('Loss_2'):
            matching_true_boxes, detectors_mask, loc_scale = self.preprocess_true_boxes(self.y,
                                                                                   self.anchors2,
                                                                                   tf.shape(self.yolo789),
                                                                                   self.img_shape)
            objectness_loss = self.confidence_loss(self.pred_xy2, self.pred_wh2, self.pred_confidence2, self.y, detectors_mask)
            cord_loss = self.cord_cls_loss(detectors_mask, matching_true_boxes,
                                      self.config.classes, self.pred_class_prob2, self.loc_txywh2, loc_scale)
            loss3 = objectness_loss + cord_loss
        self.loss = loss1 + loss2 + loss3
        
    def optimize(self):
        lr = tf.train.piecewise_constant(self.global_step_tensor, self.config.train.lr_steps, [1e-3, 1e-4, 1e-5]) 
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=lr,decay=0.0005).minimize(self.loss, global_step=self.global_step_tensor)
#        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Scale")
#        with tf.control_dependencies(update_op):
#            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=vars_det)
    
    def preprocess_true_boxes(self, true_boxes, anchors, feat_size, image_size):
        """
        :param true_boxes: x, y, w, h, class
        :param anchors: [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        :param feat_size:
        :param image_size:
        :return:
        """
        num_anchors = self.config.num_anchors_per_layer
    
        true_wh = tf.expand_dims(true_boxes[..., 2:4], 2)  # [batch, 30, 1, 2]
        true_wh_half = true_wh / 2.                        # [batch, 30, 1, 2]
        true_mins = 0 - true_wh_half                       # [batch, 30, 1, 2]
        true_maxes = true_wh_half                          # [batch, 30, 1, 2]
    
        img_wh = tf.reshape(tf.stack([image_size[2], image_size[1]]), [1, -1]) # [1, 2]
        anchors = anchors / tf.cast(img_wh, tf.float32)  # normalize
        anchors_shape = tf.shape(anchors)  # [num_anchors, 2]
        anchors = tf.reshape(anchors, [1, 1, anchors_shape[0], anchors_shape[1]])  # [1, 1, num_anchors, 2]
        anchors_half = anchors / 2.                                                # [1, 1, num_anchors, 2]
        anchors_mins = 0 - anchors_half                                            # [1, 1, num_anchors, 2]
        anchors_maxes = anchors_half                                               # [1, 1, num_anchors, 2]
    
        intersect_mins = tf.maximum(true_mins, anchors_mins)                # [batch, 30, num_anchors, 2]
        intersect_maxes = tf.minimum(true_maxes, anchors_maxes)             # [batch, 30, num_anchors, 2]
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)     # [batch, 30, num_anchors, 2]
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   # [batch, 30, num_anchors]
    
        true_areas = true_wh[..., 0] * true_wh[..., 1]      # [batch, 30, 1]
        anchors_areas = anchors[..., 0] * anchors[..., 1]   # [1, 1, num_anchors]
    
        union_areas = true_areas + anchors_areas - intersect_areas  # [batch, 30, num_anchors]
    
        iou_scores = intersect_areas / union_areas  # [batch, 30, num_anchors] IOU
        
        valid = tf.logical_not(tf.reduce_all(tf.equal(iou_scores, 0), axis=-1))     # [batch, 30]
        iout_argmax = tf.cast(tf.argmax(iou_scores, axis=-1), tf.int32)   # [batch, 30], (0, 1, 2)
        anchors = tf.reshape(anchors, [-1, 2])      # has been normalize by img shape
        anchors_cf = tf.gather(anchors, iout_argmax)   # [batch, 30, 2]
    
        feat_wh = tf.reshape(tf.stack([feat_size[2], feat_size[1]]), [1, -1])  # (1, 2)
        cxy = tf.cast(tf.floor(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32)),tf.int32)    # [batch, 30, 2]   bx = cx + σ(tx)
        sig_xy = tf.cast(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32) - tf.cast(cxy, tf.float32),tf.float32)   # [batch, 30, 2]
        idx = cxy[..., 1] * (num_anchors * feat_size[2]) + num_anchors * cxy[..., 0] + iout_argmax  # [batch, 30]
        idx_one_hot = tf.one_hot(idx, depth=feat_size[1] * feat_size[2] * num_anchors)   # [batch, 30, 13x13x3]
        idx_one_hot = tf.reshape(idx_one_hot,[-1, self.config.train.max_truth, feat_size[1], feat_size[2], num_anchors,1])  # (batch, 30, 13, 13, 3, 1)
        loc_scale = 2 - true_boxes[..., 2] * true_boxes[..., 3]     # (batch, 30)
        mask = []
        loc_cls = []
        scale = []
        for i in range(self.config.batch_size):
            idx_i = tf.where(valid[i])[:, 0]    # (?, )    # false / true
            mask_i = tf.gather(idx_one_hot[i], idx_i)   # (?, 13, 13, 3, 1)
    
            scale_i = tf.gather(loc_scale[i], idx_i)    # (?, )
            scale_i = tf.reshape(scale_i, [-1, 1, 1, 1, 1])     # (?, 1, 1, 1, 1)
            scale_i = scale_i * mask_i      # (?, 13, 13, 3, 1)
            scale_i = tf.reduce_sum(scale_i, axis=0)        # (13, 13, 3, 1)
            scale_i = tf.maximum(tf.minimum(scale_i, 2), 1)
            scale.append(scale_i)
    
            true_boxes_i = tf.gather(true_boxes[i], idx_i)    # (?, 5)
            sig_xy_i = tf.gather(sig_xy[i], idx_i)    # (?, 2)
            anchors_cf_i = tf.gather(anchors_cf[i], idx_i)    # (?, 2)
            twh = tf.log(true_boxes_i[:, 2:4] / anchors_cf_i)
            loc_cls_i = tf.concat([sig_xy_i, twh, true_boxes_i[:, -1:]], axis=-1)    # (?, 5)
            loc_cls_i = tf.reshape(loc_cls_i, [-1, 1, 1, 1, 5])     # (?, 1, 1, 1, 5)
            loc_cls_i = loc_cls_i * mask_i      # (?, 13, 13, 3, 5)
            loc_cls_i = tf.reduce_sum(loc_cls_i, axis=[0])  # (13, 13, 3, 5)
            # exception, one anchor is responsible for 2 or more object
            loc_cls_i = tf.concat([loc_cls_i[..., :4], tf.minimum(loc_cls_i[..., -1:], 19)], axis=-1)
            loc_cls.append(loc_cls_i)
    
            mask_i = tf.reduce_sum(mask_i, axis=[0])    # (13, 13, 3, 1)
            mask_i = tf.minimum(mask_i, 1)
            mask.append(mask_i)
    
        loc_cls = tf.stack(loc_cls, axis=0)     # (σ(tx), σ(tx), tw, th, cls)
        mask = tf.stack(mask, axis=0)
        scale = tf.stack(scale, axis=0)
        return loc_cls, mask, scale
    
    
    def confidence_loss(self, pred_xy, pred_wh, pred_confidence, true_boxes, detectors_mask):
        """
        :param pred_xy: [batch, 13, 13, 5, 2] from yolo_det
        :param pred_wh: [batch, 13, 13, 5, 2] from yolo_det
        :param pred_confidence: [batch, 13, 13, 5, 1] from yolo_det
        :param true_boxes: [batch, 30, 5]
        :param detectors_mask: [batch, 13, 13, 5, 1]
        :return:
        """
        pred_xy = tf.expand_dims(pred_xy, 4)  # [batch, 13, 13, 3, 1, 2]
        pred_wh = tf.expand_dims(pred_wh, 4)  # [batch, 13, 13, 3, 1, 2]
    
        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half
    
        true_boxes_shape = tf.shape(true_boxes)  # [batch, num_true_boxes, box_params(5)]
        true_boxes = tf.reshape(true_boxes, [
            true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
            ])  # [batch, 1, 1, 1, num_true_boxes, 5]
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]
    
        # Find IOU of each predicted box with each ground truth box.
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half
    
        intersect_mins = tf.maximum(pred_mins, true_mins)   # [batch, 13, 13, 3, 1, 2] [batch, 1, 1, 1, num_true_boxes, 2]
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)    # [batch, 13, 13, 3, num_true_boxes, 2]
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)    # [batch, 13, 13, 3, num_true_boxes, 2]
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   # [batch, 13, 13, 3, num_true_boxes, 2]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]      # [batch, 13, 13, 3, num_true_boxes]
        true_areas = true_wh[..., 0] * true_wh[..., 1]      # [batch, 13, 13, 3, 1]
        union_areas = pred_areas + true_areas - intersect_areas     # [batch, 1, 1, 1, num_true_boxes]
        iou_scores = intersect_areas / union_areas      # [batch, 13, 13, 3, num_true_boxes]
        # Best IOUs for each loction.
        best_ious = tf.reduce_max(iou_scores, axis=-1, keep_dims=True)  # Best IOU scores.  [batch, 13, 13, 3, 1]
    
        # A detector has found an object if IOU > thresh for some true box.
        object_ignore = tf.cast(best_ious > self.config.train.ignore_thresh, best_ious.dtype)
        no_object_weights = (1 - object_ignore) * (1 - detectors_mask)  # [batch, 13, 13, 5, 1]
        no_objects_loss = no_object_weights * tf.square(pred_confidence)
        objects_loss = detectors_mask * tf.square(1 - pred_confidence)
    
        objectness_loss = tf.reduce_sum(objects_loss + no_objects_loss)
        return objectness_loss
    
    
    def cord_cls_loss(self, detectors_mask,matching_true_boxes,num_classes,pred_class_prob,pred_boxes,loc_scale):
        """
        :param detectors_mask: [batch, 13, 13, 3, 1]
        :param matching_true_boxes: [batch, 13, 13, 3, 5]   [σ(tx), σ(ty), tw, th, cls]
        :param num_classes: 20
        :param pred_class_prob: [batch, 13, 13, 3, 20]
        :param pred_boxes: [batch, 13, 13, 3, 4]
        :param loc_scale: [batch, 13, 13, 3, 1]
        :return:
            mean_loss: float
            mean localization loss across minibatch
        """
    
        # Classification loss for matching detections.
        # NOTE: YOLO does not use categorical cross-entropy loss here.
        matching_classes = tf.cast(matching_true_boxes[..., 4], tf.int32)   # [batch, 13, 13, 3]
        matching_classes = tf.one_hot(matching_classes, num_classes)    # [batch, 13, 13, 3, 20]
        classification_loss = (detectors_mask * tf.square(matching_classes - pred_class_prob))   # [batch, 13, 13, 3, 20]
    
        # Coordinate loss for matching detection boxes.   [σ(tx), σ(ty), tw, th]
        matching_boxes = matching_true_boxes[..., 0:4]
        coordinates_loss = (detectors_mask * loc_scale * tf.square(matching_boxes - pred_boxes))
    
        classification_loss_sum = tf.reduce_sum(classification_loss)
        coordinates_loss_sum = tf.reduce_sum(coordinates_loss)
    
        return classification_loss_sum + coordinates_loss_sum
        
    def pedict(self, img_hw, iou_threshold=0.5, score_threshold=0.4):
        """
        follow yad2k - yolo_eval
        For now, only support single image prediction
        :param iou_threshold:
        :return:
        """
        img_hwhw = tf.expand_dims(tf.stack([img_hw[0], img_hw[1]] * 2, axis=0), axis=0)
        with tf.name_scope('Predict_0'):
            pred_loc0 = tf.concat([self.pred_xy0[..., 1:] - 0.5 * self.pred_wh0[..., 1:],
                                   self.pred_xy0[..., 0:1] - 0.5 * self.pred_wh0[..., 0:1],
                                   self.pred_xy0[..., 1:] + 0.5 * self.pred_wh0[..., 1:],
                                   self.pred_xy0[..., 0:1] + 0.5 * self.pred_wh0[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc0 = tf.maximum(tf.minimum(pred_loc0, 1), 0)
            pred_loc0 = tf.reshape(pred_loc0, [-1, 4]) * img_hwhw
            pred_obj0 = tf.reshape(self.pred_confidence0, shape=[-1])
            pred_cls0 = tf.reshape(self.pred_class_prob0, [-1, self.config.classes])
        with tf.name_scope('Predict_1'):
            pred_loc1 = tf.concat([self.pred_xy1[..., 1:] - 0.5 * self.pred_wh1[..., 1:],
                                   self.pred_xy1[..., 0:1] - 0.5 * self.pred_wh1[..., 0:1],
                                   self.pred_xy1[..., 1:] + 0.5 * self.pred_wh1[..., 1:],
                                   self.pred_xy1[..., 0:1] + 0.5 * self.pred_wh1[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc1 = tf.maximum(tf.minimum(pred_loc1, 1), 0)
            pred_loc1 = tf.reshape(pred_loc1, [-1, 4]) * img_hwhw
            pred_obj1 = tf.reshape(self.pred_confidence1, shape=[-1])
            pred_cls1 = tf.reshape(self.pred_class_prob1, [-1, self.config.classes])
        with tf.name_scope('Predict_2'):
            pred_loc2 = tf.concat([self.pred_xy2[..., 1:] - 0.5 * self.pred_wh2[..., 1:],
                                   self.pred_xy2[..., 0:1] - 0.5 * self.pred_wh2[..., 0:1],
                                   self.pred_xy2[..., 1:] + 0.5 * self.pred_wh2[..., 1:],
                                   self.pred_xy2[..., 0:1] + 0.5 * self.pred_wh2[..., 0:1]
                                   ], axis=-1)  # (y1, x1, y2, x2)
            pred_loc2 = tf.maximum(tf.minimum(pred_loc2, 1), 0)
            pred_loc2 = tf.reshape(pred_loc2, [-1, 4]) * img_hwhw
            pred_obj2 = tf.reshape(self.pred_confidence2, shape=[-1])
            pred_cls2 = tf.reshape(self.pred_class_prob2, [-1, self.config.classes])

        self.pred_loc = tf.concat([pred_loc0, pred_loc1, pred_loc2], axis=0, name='pred_y1x1y2x2')
        self.pred_obj = tf.concat([pred_obj0, pred_obj1, pred_obj2], axis=0, name='pred_objectness')
        self.pred_cls = tf.concat([pred_cls0, pred_cls1, pred_cls2], axis=0, name='pred_clsprob')

        # score filter
        box_scores = tf.expand_dims(self.pred_obj, axis=1) * self.pred_cls      # (?, 20)
        box_label = tf.argmax(box_scores, axis=-1)      # (?, )
        box_scores_max = tf.reduce_max(box_scores, axis=-1)     # (?, )

        pred_mask = box_scores_max > score_threshold
        boxes = tf.boolean_mask(self.pred_loc, pred_mask)
        scores = tf.boolean_mask(box_scores_max, pred_mask)
        classes = tf.boolean_mask(box_label, pred_mask)

        # non_max_suppression
        idx_nms = tf.image.non_max_suppression(boxes, scores,max_output_size=5,iou_threshold=iou_threshold)
        boxes = tf.gather(boxes, idx_nms)
        scores = tf.gather(scores, idx_nms)
        classes = tf.gather(classes, idx_nms)

        return boxes, scores, classes
        