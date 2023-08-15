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
# from train import Batch_Size,Input_Size
#transform 是对图像进行预处理、数据增强。compose将多个处理步骤整合到一起
#ToTensor:将原始取值0-255像素值，归一化伪0-1
#Normalize：用像素值的均值和标准偏差对像素值进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.465,0.406],std=[0.229,0.224,0.225])
])

#主要采用N 位寄存器对N个状态进行编码，每个状态都有它独立的寄存器位，并且再任意时候只有一位有效。
# 此编码是分类变量作为二进制向量的表示。这首先要求分类值映射到整数值，然后每个整数值被表示成二进制向量
def onehot(data,n):
    buf = np.zeros(data.shape+(n,))
    nmsk = np.arange(data.size)*n+data.ravel()#ravel将多维数组降为一维，且降维后可以改变原变量的值
    buf.ravel()[nmsk-1]=1
    return buf

class BagDataset(Dataset):
    def __init__(self,transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('./dataset/aeroscapes/aeroscapes/JPEGImages'))   #地址
        # return len(os.listdir('./dataset/bulid/src1'))   #地址

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
        #读取原图
        imgA_name = os.listdir('./dataset/aeroscapes/aeroscapes/JPEGImages')[idx]
        # imgA_name = os.listdir('./dataset/bulid/src1')[idx]
        imgB_name = os.listdir('./dataset/aeroscapes/aeroscapes/SegmentationClass')[idx]
        # imgB_name = os.listdir('./dataset/bulid/gt1')[idx]
        imgA = cv2.imread('./dataset/aeroscapes/aeroscapes/JPEGImages/'+imgA_name)
        # imgA = cv2.imread('./dataset/bulid/src1/'+imgA_name)
        imgA = cv2.resize(imgA,Input_Size)

        #读取标签图
        imgB =cv2.imread('./dataset/aeroscapes/aeroscapes/SegmentationClass/'+imgB_name)
        # imgB =cv2.imread('./dataset/bulid/gt1/'+imgB_name)
        imgB = cv2.resize(imgB, Input_Size)

        # imgA,imgB = self.get_rand_data(imgA,imgB)#用于数据增强


        label = imgB[:,:,0]
        imgB = label


        if self.transform:
            imgA = self.transform(imgA)#一转成向量后，imgA通道就变成（C,H,W）
        return imgA,imgB

bag = BagDataset(transform)
train_size = int(0.9*len(bag))
test_size = len(bag)-train_size

train_dataset,test_dataset = random_split(bag,[train_size,test_size])#按照上述比例（9：1）划分训练集和测试集
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