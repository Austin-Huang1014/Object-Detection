#!/usr/bin/env python

import os
import h5py
import  numpy as np
from PIL import Image
import cv2

def write_hdf5(image,outfile):
    with h5py.File(outfile,'w') as f:      
        f['YB_image'] = image

image_test = '/home/austin/Test/src/joy_control/data/dataset'
dataset_path = '/home/austin/Test/src/joy_control/data/'



Nimgs = 8
channels = 3
height = 256
weight = 256

def get_datasets(image_test):
    imgs = []
    i = 0
    for filename in os.listdir(r"/"+image_test):
        image_path = os.path.join(image_test, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (weight, height), interpolation=cv2.INTER_AREA)
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        i += 1
        print(i)
    return imgs

test_image = get_datasets(image_test)
print('save')
write_hdf5(test_image,dataset_path+'cov_v1.hdf5')
print('game over')
