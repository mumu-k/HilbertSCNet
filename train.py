#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/3 21:47
# @Author : Xu Linkang
# @Site : 
# @File : train.py
# @Software: PyCharm
from  datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import test_dataloader,train_dataloader,train_size,test_size,Batch_Size,Input_Size
from Net.efficient_new import efficientNet
from Net.efficientv2_new import efficient_v2
from IoU import IOUMetric
import pandas as pd
from tqdm import tqdm
from Loss.CE_Loss import CE_Loss

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
num_classes = 2
IOU = IOUMetric(num_classes)
epoch = 30
# alpha =1


def train(epo_num=50,show_vgg_params=False):

    efficient = efficient_v2(num_classes,pretrained=True)
    net = efficient.to(device)
    
    optimizer = optim.Adam(net.parameters(),lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    Train_Loss = []
    Test_Loss = []
    miou = []
    acc = []
    ciou = []
    best_miou = 0

    for epo in range(epo_num):
        train_loss = 0
        net.train()
        print('start Trian!')
        with tqdm(total=train_size//Batch_Size,desc=f'Epoch{epo+1}/{epo_num}',postfix=dict) as pbar:
            for _,(img,mask) in enumerate(train_dataloader):
                img = img.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                output = net(img)
                loss = CE_Loss(output,mask.long(),num_classes=num_classes)
                loss.backward()
                optimizer.step()
                iter_loss = loss.item()
                train_loss += iter_loss
                pbar.set_postfix(**{'train_loss':train_loss/len(train_dataloader)})
                pbar.update(1)

        test_loss = 0
        MIoU = 0
        Acc = 0
        CIoU = 0
        net.eval()
        print('start Eval!')
        with tqdm(total=test_size//Batch_Size, desc=f'Epoch{epo+1}/{epo_num}', postfix=dict) as pbar:
            with torch.no_grad():
                for _,(img,mask) in enumerate(test_dataloader):
                    img = img.to(device)
                    mask = mask.to(device)
                    optimizer.zero_grad()
                    output = net(img)
                    loss = CE_Loss(output,mask.long(),num_classes=num_classes)
                    iter_loss = loss.item()     
                    test_loss += iter_loss

                    output = torch.softmax(output,dim=1)
                    output_np = output.cpu().numpy().copy()
                    output_np = np.argmax(output_np,axis = 1)
                    mask_np = mask.cpu().numpy().copy()

                    _,test_Acc,Class_IoU,test_MIoU,_ = IOUMetric.evaluate(IOU,output_np,mask_np)
                    iter_test_Acc = test_Acc.item()
                    iter_test_MIoU = test_MIoU.item()

                    MIoU += iter_test_MIoU
                    Acc += iter_test_Acc
                    CIoU += Class_IoU
                    pbar.set_postfix(**{'test_loss':test_loss/len(test_dataloader),
                                        'miou':MIoU/len(test_dataloader),
                                        'acc':Acc/len(test_dataloader)})
                    pbar.update(1)
        eval_miou = MIoU / len(test_dataloader)
        if best_miou < eval_miou:
            torch.save(net,'./results/model.pth')
            best_miou = eval_miou
            print('save success!')
        lr_scheduler.step()

        Train_Loss.append(train_loss/len(train_dataloader))
        Test_Loss.append(test_loss/len(test_dataloader))
        miou.append(MIoU/len(test_dataloader))
        acc.append(Acc/len(test_dataloader))
        ciou.append(CIoU/len(test_dataloader))


    data_log = pd.DataFrame({'train_loss':Train_Loss,
                             'test_loss':Test_Loss,
                             'miou':miou,
                             'acc':acc,
                             'ciou':ciou})
    data_log.to_csv("./log/log_batch{}_epoch{}_HW{}.csv".format(Batch_Size,epoch,Input_Size),index=True)
    print('Ending!!!')


if __name__ == '__main__':
    train(epo_num=epoch,show_vgg_params=False)
