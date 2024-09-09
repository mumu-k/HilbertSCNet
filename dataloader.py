#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/2 15:43
# @Author : Xu Linkang
# @Site : 
# @File : dataloader.py
# @Software: PyCharm
import os
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import numpy as np
import cv2
import random
from PIL  import Image


Batch_Size = 4
Input_Size = (512,512)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.465,0.406],std=[0.229,0.224,0.225])
])

# The N-bit registers are used to encode the N states, each state has its own register bit and only one bit is valid at any given time.
# This encoding is a representation of the categorical variables as binary vectors. This first requires the categorical values to be mapped to integer values, and then each integer value is represented as a binary vector
def onehot(data,n):
    buf = np.zeros(data.shape+(n,))
    nmsk = np.arange(data.size)*n+data.ravel()#ravel reduces multi-dimensional arrays to one dimension and can change the value of the original variable after reduction
    buf.ravel()[nmsk-1]=1
    return buf

class BagDataset(Dataset):
    def __init__(self,transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('./dataset/aeroscapes/aeroscapes/JPEGImages'))   #Address of the dataset

    def get_rand_data(self,image,label):
        flip = rand()<.5
        Rotation = rand()<.5
        Rotation_angle_list = [90,180,270]

        flip_code = int(((1-(-1)) * np.random.random() + (-1))*10)
        if flip:
            image = cv2.flip(image,flip_code)
            label = cv2.flip(label,flip_code)

        if Rotation:
            angle = Rotation_angle_list[random.randrange(len(Rotation_angle_list))]
            image = Rotate(image,angle)
            label = Rotate(label,angle)

        return image,label

    def __getitem__(self, idx):
        #imgA-original figure，imgB-mask
        imgA_name = os.listdir('./dataset/aeroscapes/aeroscapes/JPEGImages')[idx]
        imgB_name = os.listdir('./dataset/aeroscapes/aeroscapes/SegmentationClass')[idx]
        
        #Read original image
        imgA = cv2.imread('./dataset/aeroscapes/aeroscapes/JPEGImages/'+imgA_name)
        imgA = cv2.resize(imgA,Input_Size)

        #Read masks
        imgB =cv2.imread('./dataset/aeroscapes/aeroscapes/SegmentationClass/'+imgB_name)
        imgB = cv2.resize(imgB, Input_Size)

        label = imgB[:,:,0]
        imgB = label


        if self.transform:
            imgA = self.transform(imgA)
        return imgA,imgB

bag = BagDataset(transform)
train_size = int(0.9*len(bag))
test_size = len(bag)-train_size

train_dataset,test_dataset = random_split(bag,[train_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=Batch_Size,shuffle=True,num_workers=0,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=Batch_Size,shuffle=True,num_workers=0,drop_last=True)


def rand(a = 0, b =1):
    return np.random.rand()*(b-a) + a

def Rotate(image, angle=15, scale=1):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image




if __name__ == '__main__':

    for train_batch in train_dataloader:
        print(train_batch)
    for test_batch in test_dataloader:
        print(test_batch)
    # print(train_lines)
