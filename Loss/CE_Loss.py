#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/1 12:37
# @Author : Xu Linkang
# @Site : 
# @File : CE_Loss.py
# @Software: PyCharm
import  scipy.signal
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

def CE_Loss(inputs,target,num_classes=21):
    # print(inputs.shape)
    n,c,h,w = inputs.size()
    nt,ht,wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs,size=(ht,wt),mode='bilinear',align_corners=True)

    temp_inputs = inputs.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    temp_target = target.view(-1)

    CE_Loss = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs,dim=-1),temp_target)
    return CE_Loss