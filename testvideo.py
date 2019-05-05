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
import cv2


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

def predict_img(img,net,thresd = 0.995):
    try:
        image = loader(img).unsqueeze(0)
        image.to(device, torch.float)
        net.eval()
        with torch.no_grad():
            output = net(image)
            sout = F.softmax(output,dim=1)
            _,predicted = output.max(1)
            prefloat = float(sout[-1,1])
            if predicted[0] == 1 and prefloat>=thresd:
                print('is person : %0.5f'%prefloat)
                return 1, prefloat
            else:
                #print("no person")
                return 0, prefloat

    except:
        print("failed ")
        return -1,0



def video():
    net = MobileNetV2(num_classes=2)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    net = net.to(device)
    net = loadmodel(net)

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()
        #frame = cv2.imread('image/29.png')
        cv2.imshow('tst', frame)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        #frame = Image.open('image/1.jpg').convert('RGB')
        #frame.show('dd')
        person, pre = predict_img(frame,net,thresd=0.9)
        if person == 1:
            print('  ')
        else:
            print('no person : %0.5f '%pre)

        cv2.waitKey(10)

video()