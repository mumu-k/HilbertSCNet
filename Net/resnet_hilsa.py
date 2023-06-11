from Net.resnet import resnet50
from test.Hil_SA import Hilbert_Sa
import torch
import torch.nn as nn

import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self,dilate_scale=16,pretrained = True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)
        #--------------------------------------------------------------------------#
        #根据下采样因子修改卷积的步长与膨胀系数
        #当downsample=16时，我们最终获得两个特征层，shape分别是：30，30，1024和30，30，2048
        #--------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate,dilate=2))
            model.layer4.apply(partial(self._nostride_dilate,dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate,dilate=2))

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


    def _nostride_dilate(self,m,dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2,2):
                m.stride = (1,1)
                if m.kernel_size ==(3,3):
                    m.dilation = (dilate//2,dilate//2)
                    m.padding = (dilate//2,dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # print(x.shape)
        x = self.layer1(x)

        # x = self.se1(x)
        # print(x.shape)
        x = self.layer2(x)


        # x = self.se2(x)
        # print(x.shape)
        x_aux = self.layer3(x)

        # x_aux = self.se3(x_aux)
        # print(x_aux.shape)
        x = self.layer4(x_aux)

        # x = self.se4(x)
        # print(x.shape)

        return x


class res_hilsa(nn.Module):
    def __init__(self,num_class,pretrained = False):
        self.backbone = Resnet()