# File Description: This file is used to check and pickle the data

import numpy as np
import os.path
from PIL import Image
import cv2
import pickle
import json
import utils
import random

class LoadData:
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.trainClass = list()
        self.valClass = list()
        self.trainSmooth = list()
        self.valSmooth = list()
        self.noise_type='sym'


        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            textFile = json.load(textFile)
            random.shuffle(textFile)
            for line in textFile:
                img_file = self.data_dir+'/'+line['name']
                if trainStg:
                    self.trainImList.append(img_file)
                    self.trainClass.append(line['label'])
                    # self.trainSmooth.append(line['smooth_label'])
                else:
                    self.valImList.append(img_file)
                    self.valClass.append(line['label'])
                    # self.valSmooth.append(line['smooth_label'])
        return 0


    def processData(self):
        print('Processing training data')
        return_val = self.readFile('train_new2.json', True)

        print('Processing validation data')
        return_val1 = self.readFile('test_new2.json')

        print('Pickling data')
        if return_val ==0 and return_val1 ==0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainClass'] = self.trainClass
            data_dict['valIm'] = self.valImList
            data_dict['valClass'] = self.valClass
            if self.noise_type is not None:
                self.trainClass=np.asarray([[self.trainClass[i]] for i in range(len(self.trainClass))])
                self.valClass=np.asarray([[self.valClass[i]] for i in range(len(self.valClass))])
                train_noisy_labels, actual_noise_rate = utils.noisify(dataset='colon', train_labels=self.trainClass, noise_type='symmetric', noise_rate=0.4, random_state=0, nb_classes=self.classes)
                train_noisy_labels=[i[0] for i in train_noisy_labels]
                data_dict['trainNoiseClass'] = train_noisy_labels
                val_noisy_labels, actual_noise_rate = utils.noisify(dataset='colon', train_labels=self.valClass, noise_type='symmetric', noise_rate=0.4, random_state=0, nb_classes=self.classes)
                val_noisy_labels=[i[0] for i in val_noisy_labels]
                data_dict['valNoiseClass'] = val_noisy_labels
            # data_dict['trainSmooth'] = self.trainSmooth
            # data_dict['valSmooth'] = self.valSmooth
            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            print(len(data_dict['trainIm']),len(data_dict['valIm']))
            return data_dict
        else:
            return None



