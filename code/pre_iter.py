import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
import time
import datetime
import os
import sys
from PIL import Image,ImageOps
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import re
from tqdm import tqdm
import math
import pandas as pd
import csv
import json
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class classify:
    def __init__(self,img_path,model_path, save_path, num_classes):
        self.model_path = model_path
        self.image_path = image_path
        self.save_path = save_path
        self.num_classes = num_classes
        self.test_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def load_net(self):
        state_dict = torch.load(self.model_path)
        loaded_model = state_dict['model']
        net = torchvision.models.resnet34(num_classes=self.num_classes)
        net.eval()
        net.load_state_dict(loaded_model)
        net.to('cuda')
        return net

    def predict(self,net,patch):
        imgblob = self.test_transforms(patch).unsqueeze(0).to("cuda")
        imgblob = Variable(imgblob)
        output = net(imgblob)
        probability=F.softmax(output,dim=1)
        _,label = torch.max(probability,1)
        return label,probability[0,label]

    def run(self):
        with torch.no_grad():
            net = self.load_net()
            pre_labels = []
            pre_scores = []
            for img_path in tqdm(self.image_path):
                img = Image.open(img_path)
                label,score = self.predict(net,img)
                pre_labels.append(label.cpu()[0].numpy().tolist())
                pre_scores.append(score.cpu().numpy().tolist()[0])
        return pre_labels, pre_scores

def meanStr(line):
    num_list = json.loads(line['logits'])[:30]
    num_list = np.log10(num_list)
    return np.mean(num_list)

if __name__ == '__main__':
    record = pd.read_csv('iteration/bishe/digest_20_1.csv')
    record['meanProbs'] = record.apply(meanStr,axis=1)
    thred = record.sort_values(by=['meanProbs'])['meanProbs'].iloc[int(len(record)*0.2)]
    noise_candidate = record[record['meanProbs']<thred].copy()
    print(len(noise_candidate))
    image_path = noise_candidate['name'].tolist()
    # data = pickle.load(open('pickle_data/colon_20.p', "rb"))
    # image_path = data['valIm']
    model_path = 'result/bishe/digest_20_1/model_58.pth'
    save_path = 'iteration/bishe/digest_20_1_correct.csv'
    classifier = classify(image_path,model_path,save_path,2)
    pre_labels,pre_scores = classifier.run()
    noise_candidate['correct_label']=pre_labels
    noise_candidate['correct_scores']=pre_scores
    noise_candidate.to_csv(save_path, index=False)
    print(np.sum((noise_candidate['y'].values==np.array(pre_labels))),len(noise_candidate))