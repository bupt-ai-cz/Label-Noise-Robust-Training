# File Description: This file is used to create data tuples

import cv2
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import json

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, smoothList=[], transform=None):
        self.imList = imList
        self.labelList = labelList
        # self.smoothList = smoothList
        self.transform = transform

    def __len__(self):
        return len(self.imList)
 
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        ##noise identify
        # data = np.array(json.loads(self.imList[idx]))
        # data = torch.from_numpy(data[np.newaxis, :]).float()
        # label = self.labelList[idx]
        # return data,label,idx
        ######################
        image_path = self.imList[idx]
        image = self.pil_loader(image_path)
        label_name = self.labelList[idx]
        ########################
        # smooth_label = self.smoothList[idx]
        # smooth_label = torch.from_numpy(np.array([1-smooth_label,smooth_label])).float()
        # image = cv2.imread(image_name)
        # image = cv2.resize(image,(224,224))
        # label = cv2.imread(label_name, 0)
        # label = cv2.resize(label,(576,576))
        # _,label = cv2.threshold(label,127,1,cv2.THRESH_BINARY)
        ###############
        if self.transform:
            image= self.transform(image)
        return image, label_name, idx