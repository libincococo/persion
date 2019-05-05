import os,sys,io,re
import random
import numpy as np
import random
import cv2
import pandas as pd
from pandas import Series, DataFrame

data_path = '/Volumes/WD/libin/DTA_Person/'
train_path = {('train_pos1',1),
              ('train_pos3',1),
              ('train_pos4',1),
              ('train_pos5',1),
              ('train_neg1',0),
              ('train_neg2',0),
              ('train_neg4',0)
            }


test_path = {('test_pos1',1),
              ('test_pos2',1),
              ('test_neg1',0)
             }

def read_file(path,rootpath):
    files = os.listdir(os.path.join(rootpath,path))
    return files

def write_txt(filename, pathlist,rootpath):
    filelists = []
    flagelist = []
    for dirs,flage in pathlist:
        print(flage)
        lists = read_file(dirs,rootpath)
        #filelists += lists
        #for step in lists:
        #    filelists.append(dirs+"/"+step)
        filelists += [dirs+'/'+step for step in lists]
        flagelist += [flage for step in lists]

    datas = {'filename':filelists,"classes":flagelist}
    df = DataFrame(datas)

    print(len(filelists))
    df.to_csv(filename,index=False,header=False)

def test_csv(filename,path):
    data = pd.read_csv(filename)
    for idx, item in data.iterrows():

        #print(item[0])
        #print(item[1])
        file = os.path.join(path,item[0])
        if os.path.isfile(file) == False:
            print("the file is not exist ",file)

write_txt('train.cvs',train_path,data_path)
write_txt('test.cvs',test_path,data_path)
test_csv('train.cvs',data_path)
test_csv('test.cvs',data_path)


