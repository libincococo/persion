from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from persionDataSet import persiondataset
from PIL import Image


import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

imagesize = 56

loader = transforms.Compose([
    transforms.Resize((imagesize,imagesize)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

def loadmodel(net):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cpu':
        checkpoint = torch.load('./checkpoint/ckpt.t7',map_location='cpu')
        net.load_state_dict(checkpoint['net'],strict=False)
    else:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'],strict=False)
    return net

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def predict(filename,net,thresd = 0.995):
    #image = Image.open(filename).convert('RGB')
    try:
        image = image_loader(filename)
        net.eval()
        with torch.no_grad():

            output = net(image)
            sout = F.softmax(output,dim=1)
            _,predicted = output.max(1)
            prefloat = float(sout[-1,1])

            #if predicted[0] == 1 and prefloat>=thresd:
            if  prefloat >= thresd:
                print(filename,'is person :',prefloat)
                return 1
            else:
                return 0

    except:
        print("failed ",filename)
        return -1

#predict("./image/4.jpg")

def evalpath():
    root = '/media/sky/8C6C1B5A6C1B3E80/dataset/DTA_Person/test_classes/dog'
    #root = '/media/sky/8C6C1B5A6C1B3E80/dataset/DTA_Person/train_neg4'
    root = "image"
    files = os.listdir(root)
    net = MobileNetV2(num_classes=2)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    net = net.to(device)
    net = loadmodel(net)
    n_person = 0
    for step, file in enumerate(files):
        if predict(os.path.join(root,file),net,thresd=0.0) == 1:
            n_person = n_person+1
    print('acc is :',float(n_person)/float(step),' total num is :',step)

evalpath()
