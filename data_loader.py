#coding=utf-8

import cv2
import copy
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

class PascalVOCDataset(Dataset):

    def __init__(self, train_file_list):
        self.train_list = []
        with open(train_file_list) as F:
            for i in F:
                self.train_list.append(i.rstrip())

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):

        X = np.array(Image.open('datasets/images/'+self.train_list[item]+'.jpg')) / 255.

        boxes = self.load_pascal_annotation('datasets/annotations/'+self.train_list[item]+'.xml')

        return X, boxes, self.train_list[item]

    def load_pascal_annotation(self, filename):
        tree = ET.parse(filename)
        objs = tree.findall('object')
        objs = [x for x in objs if x.find('name').text == 'stomap']
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 5))
		
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes[ix, :] = [x1, y1, x2, y2, 1]

        return boxes

class KFDataseteval(Dataset):

    def __init__(self, train_file_list):
        self.train_list = []
        with open(train_file_list) as F:
            for i in F:
                self.train_list.append(i.rstrip())

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):
        cvimage = cv2.imread('images/'+self.train_list[item]+'.jpg')
        X = np.array(Image.open('images/'+self.train_list[item]+'.jpg'))

        boxes = self.load_pascal_annotation('annotations/'+self.train_list[item]+'.xml')
        X_input = np.zeros((len(boxes), 3, 96, 96))
        X_dH = np.zeros(len(boxes))
        X_dW = np.zeros(len(boxes))

        for i in range(len(boxes)):
            if boxes[i, 2] - boxes[i, 0] < 95:
                boxes[i, 2] = boxes[i, 0] + 95
                #print('*******')
            if boxes[i, 3] - boxes[i, 1] < 95:
                boxes[i, 3] = boxes[i, 1] + 95
                #print('*******')

        for i in range(len(boxes)):
            X_this = X[boxes[i,1]-1:boxes[i,3], boxes[i,0]-1:boxes[i,2], :]
            H,W,_ = X_this.shape
            dH = (H-96)//2
            dW = (W-96)//2
            X_crop = X_this[dH:dH+96, dW:dW+96, :].transpose(2, 0, 1)
            X_input[i] = X_crop / 255.
            X_dH[i] = dH
            X_dW[i] = dW
        return X_input, boxes, X_dH, X_dW, cvimage

    def load_pascal_annotation(self, filename):
        tree = ET.parse(filename)
        objs = tree.findall('object')
        objs = [x for x in objs if x.find('name').text == 'stomap']
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int32)
		
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes[ix, :] = [x1, y1, x2, y2]
        return boxes
