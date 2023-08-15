#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/9 20:18
# @Author : Xu Linkang
# @Site : 
# @File : torch_predict.py
# @Software: PyCharm
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

#图片路径
img_file = r'G:\XLK\new_model\dataset\aeroscapes\aeroscapes\JPEGImages\041002_031.jpg'

output = r'./forecast'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#######加载权重刀内存中
weight_path = r'./results/141hyperseg.pth'
model = torch.load(weight_path)
########数据集的均值和方差
MEAN = [0.485,0.465,0.406]
STD = [0.229,0.224,0.225]
normalize = transforms.Normalize(MEAN,STD)
to_tensor = transforms.ToTensor()
model.to(device)
model.eval()
# palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
#                     (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
#                     (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
# palette = [(0, 0, 0),(192,128,128),(0,128,0),(128,128,128),(128,0,0),(0,0,128),(192,0,128),(192,0,0),(192,128,0),
#            (0,64,0),(128,128,0),(0,128,128)]
palette = [(192,128,128),(0,0,255)]


def save_images(mask,image,output_path,image_file,palette,num_classes,image_h,image_w):
# def save_images(mask,output_path,image_file,palette,num_classes,image_h,image_w):

    image_file = os.path.basename(image_file).split('.')[0] #basename返回图片名字
    colorized_mask = cam_mask(mask,palette,num_classes,image_h,image_w)
    ############此部分为将预测图与原图融合#################
    colorized_mask = np.asarray(colorized_mask)
    colorized_mask = cv2.addWeighted(image,1,colorized_mask,0.7,0)
    scr = os.path.join(output_path,image_file+'.png')
    cv2.imwrite(scr,colorized_mask)
    ######################################################
    ##############若不融合则把image形参删除，把下面代码取消注释##############
    # colorized_mask.save(os.path.join(output_path,image_file+'.png'))


def cam_mask(mask,palette,n,image_h,image_w):
    # seg_img = np.zeros((np.shape(mask)[0],np.shape(mask)[1],3))
    seg_img = np.zeros((mask.shape[0],mask.shape[1],3))
    print(seg_img.shape)
    for c in range(n):
        seg_img[:,:,0]+=((mask[:,:]==c)*(palette[c][0])).astype('uint8')
        seg_img[:,:,1]+=((mask[:,:]==c)*(palette[c][1])).astype('uint8')
        seg_img[:,:,2]+=((mask[:,:]==c)*(palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img)).resize((image_h,image_w),Image.BICUBIC)
    # colorized_mask = np.uint8(seg_img).resize((image_h,image_w),Image.NEAREST)
    return colorized_mask

image = cv2.imread(img_file)
yuan_image = image
w,h = image.shape[0],image.shape[1]
image = cv2.resize(image,(512,512))
input = normalize(to_tensor(image)).unsqueeze(0)#将图像做归一化，并增加一个维度（batch）便于放入网络
prediction = model(input.to(device))
prediction = prediction.squeeze(0).detach().cpu().numpy()#将预测结果拿去batch维度，转换到cpu上
prediction = np.argmax(prediction,axis=0)
# image_mask = save_images(prediction,output,img_file,palette,num_classes=2,image_h=h,image_w=w)
image_mask = save_images(prediction,yuan_image,output,img_file,palette,num_classes=2,image_h=h,image_w=w)


