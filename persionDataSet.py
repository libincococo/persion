import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from pandas import Series, DataFrame
import os
import argparse
import cv2
from PIL import Image

class persiondataset(Dataset):
    def __init__(self,filename,root,transform=None):
        #super.__init__(super)
        self.data = pd.read_csv(filename)
        self.rootpath = root
        self.transform = transform
        #print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = os.path.join(self.rootpath, self.data.iloc[index, 0])
        #image = cv2.imread(img_name)
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        landmarks = self.data.iloc[index,1]
        sample = {'image': image, 'classes': landmarks}

        if self.transform:
            #sample. = self.transform(sample)
            try:
                #print(image)
                image = self.transform(image)
            except:
                print(img_name)
                print(image)

        sample = {'image': image, 'classes': landmarks}

        return image, landmarks


def test():

    dataset = persiondataset('test.cvs','/Volumes/WD/libin/DTA_Person/')
    print(dataset.__len__())
    sames = dataset.__getitem__(1)

