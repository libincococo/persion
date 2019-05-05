import os,sys,io,re
import random
import numpy as np
import random
import cv2
import pandas as pd
from pandas import Series, DataFrame
import xml.etree.ElementTree as ET

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

pic_labels ="filename"
object_labels = 'object'
object_name = 'name'
object_rect_labels='bndbox'
object_xmin = 'xmin'
object_ymin = 'ymin'
object_xmax = 'xmax'
object_ymax = 'ymax'

def readxml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    #pic_filename
    object_list = []
    for child in root:
        #print(child.tag,child.text)
        if child.tag == pic_labels:
            pic_filename=child.text

        if child.tag == object_labels:
            for sub in child:
                xmin =0
                xmax =0
                ymin =0
                ymax =0
                if sub.tag == object_name:
                    classes_name = sub.text
                if sub.tag == object_rect_labels:
                    for third in sub:
                        #xmin,xmax,ymin,ymax=0
                        if third.tag == object_xmin:
                            xmin = int(third.text)
                        if third.tag == object_ymin:
                            ymin = int(third.text)
                        if third.tag == object_xmax:
                            xmax = int(third.text)
                        if third.tag == object_ymax:
                            ymax = int(third.text)
                    object_list.append((classes_name,xmin,ymin,xmax,ymax))

    #print(pic_filename,object_list)
    return pic_filename,object_list

def writepic(root,filename,object,img,step):
    rootpath = root+'/'+object
    file = root+'/'+object+'/'+filename+'_'+object+'_'+str(step)+'.jpg'
    print(file)
    cv2.imwrite(file,img)

def makeoutpic(root,picobject,dst_path='./classes'):
    file = picobject[0]
    object_list = picobject[1]

    path = os.path.join(root,file)
    if os.path.isfile(path) == False:
        print('not find file:',path)
        return None
    img = cv2.imread(path)
    #cv2.imshow('test',img)
    #cv2.waitKey(2000)

    file_b = os.path.splitext(file)[0]


    for step, object in enumerate(object_list):
        object_name = object[0]
        xmin = object[1]
        ymin = object[2]
        xmax = object[3]
        ymax = object[4]

        #object_roi = img[xmin:xmax,ymin:ymax]
        object_roi = img[ymin:ymax,xmin:xmax]
        #cv2.imshow(object_name,object_roi)
        #cv2.waitKey(2000)
        writepic(dst_path,file_b,object_name,object_roi,step)

def makerootdir(root):
    rootpath = root+'/'
    for dir in classes:
        try:
            os.mkdir(rootpath+dir)
        except:
            print('dir:',rootpath+dir,"haved ")

def make(file_xml_path,file_pic_path,file_pic_out):
    makerootdir(file_pic_out)

    for file in os.listdir(file_xml_path):
        #print(os.path.splitext(file)[1])
        if os.path.splitext(file)[1] == '.xml':
            #print(file_xml_path+file)
            try:
                object = readxml(file_xml_path+file)
                makeoutpic(file_pic_path,object,file_pic_out)
            except:
                print('do file is error :',file_xml_path+file)


make('/media/sky/8C6C1B5A6C1B3E80/dataset/voc/VOCdevkit/VOC2007TEST/Annotations/',
     '/media/sky/8C6C1B5A6C1B3E80/dataset/voc/VOCdevkit/VOC2007TEST/JPEGImages/',
     '/media/sky/8C6C1B5A6C1B3E80/dataset/DTA_Person/test_classes/')
#makerootdir("./classes")
#testojbect =readxml('/Volumes/WD/libin/VOCdevkit/VOC2007/Annotations/000033.xml')
#makeoutpic("/Volumes/WD/libin/VOCdevkit/VOC2007/JPEGImages",testojbect)
