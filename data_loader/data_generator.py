# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""
from __future__ import division

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os


class DataGenerator:
    '''Generate data'''
    def __init__(self, config):
        self.config = config
        self.get_filenames(config.xml_path)
        
    def next_batch(self):
        '''
            Batch Stochastic gradient descent
        '''
        idx = np.random.randint(0,self.config.trained_img_num,self.config.batch_size)
        yield self.load_images(idx), self.load_labels(idx)
            
    def get_filenames(self, path):
        self.filenames = [ f.split('.')[0] for f in os.listdir( path ) if os.path.isfile( os.path.join( path, f ) )]
            
    def load_images(self,idx):
        if self.config.train.random:
            self.config.train.image_resized = np.random.choice(self.config.train.image_ranged,size=1)
        image_names = [os.path.join(self.config.imgs_path,self.filenames[i])+'.jpg' for i in idx]
        image_lists = []
        for image_name in image_names:
            image_lists.append(self.convert_img(image_name))
        return np.array(image_lists)
        
    def convert_img_padding(self,image_id):
        """
            resize image with unchanged aspect ratio using padding
            ......
        """
        image = Image.open(image_id)
        image_w, image_h = image.size
        new_w = int(image_w * min(self.config.train.image_resized/image_w, self.config.train.image_resized/image_h))
        new_h = int(image_h * min(self.config.train.image_resized/image_w, self.config.train.image_resized/image_h))
        resized_image = image.resize((new_w, new_h), Image.BICUBIC)
        boxed_image = Image.new('RGB', (self.config.image_resized,self.config.image_resized), (128, 128, 128))
        boxed_image.paste(resized_image, ((self.config.image_resized-new_w)//2, (self.config.image_resized-new_h)//2))
        image_data = np.array(boxed_image, dtype='float32')/255
        return image_data
        
    def convert_img(self,image_id):
        image = Image.open(image_id)
        resized_image = image.resize((self.config.train.image_resized, self.config.train.image_resized), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')/255.0
        return image_data
        
    def load_labels(self,idx):
        label_names = [os.path.join(self.config.xml_path,self.filenames[i])+'.xml' for i in idx]
        label_lists = []
        for label_name in label_names:
            label_lists.append(self.convert_annotation(label_name))
        return np.array(label_lists)
        
    def convert_annotation(self, xml_id):
        in_file = open(xml_id)
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        bboxes=[]
        for i,obj in enumerate(root.iter('object')):
            if i >= self.config.train.max_truth:
                break
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.config.names or int(difficult) == 1:
                continue
            cls_id = self.config.names.index(cls)
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = self.convert((w, h), box) + [cls_id]
            bboxes.append(bbox)
        if len(bboxes) < self.config.train.max_truth:
            bboxes += [[0,0,0,0,0]] * (self.config.train.max_truth - len(bboxes))
        return np.array(bboxes)
        
    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return [x, y, w, h]
        