import torch
import torch.nn as nn
from Net.efficientnet_v2 import efficientv2
from test.HP_sa import Hilbert_Sa
import torch.nn.functional as F
import math

class efficient_v2(nn.Module):
    def __init__(self,num_class,pretrained = False):
        super(efficient_v2, self).__init__()
        model = efficientv2(pretrained=pretrained)
        blocks_repeat = [3,8,13,20,34,52,57]
        self.cov = nn.Conv2d(in_channels=2048, out_channels=num_class, kernel_size=1)
        self.stem = model.stem
        stage1 = []
        for i in range(blocks_repeat[0]):
            stage1.append(model.blocks[i])
        self.stage1 = nn.Sequential(*stage1)
        stage2 = []
        for i in range(blocks_repeat[0],blocks_repeat[1]):
            stage2.append(model.blocks[i])
        self.stage2 = nn.Sequential(*stage2)
        stage3 = []
        for i in range(blocks_repeat[1],blocks_repeat[2]):
            stage3.append(model.blocks[i])
        self.stage3 = nn.Sequential(*stage3)
        stage4 = []
        for i in range(blocks_repeat[2],blocks_repeat[3]):
            stage4.append(model.blocks[i])
        self.stage4 = nn.Sequential(*stage4)
        stage5 = []
        for i in range(blocks_repeat[3],blocks_repeat[4]):
            stage5.append(model.blocks[i])
        self.stage5 = nn.Sequential(*stage5)
        stage6 = []
        for i in range(blocks_repeat[4],blocks_repeat[5]):
            stage6.append(model.blocks[i])
        self.stage6 = nn.Sequential(*stage6)
        stage7 = []
        for i in range(blocks_repeat[5],blocks_repeat[6]):
            stage7.append(model.blocks[i])
        self.stage7 = nn.Sequential(*stage7)
        self.head = model.head[0]

        #######################lateral layers################
        self.latlayer1 = nn.Sequential(nn.Conv2d(1280,512,kernel_size=1,stride=1,padding=0))
        self.latlayer2 = nn.Sequential(nn.Conv2d(176,512,kernel_size=1,stride=1,padding=0))
        self.latlayer3 = nn.Sequential(nn.Conv2d(80,512,kernel_size=1,stride=1,padding=0))
        self.latlayer4 = nn.Sequential(nn.Conv2d(48,512,kernel_size=1,stride=1,padding=0))

        #####################Hil_sa layers ####################
        self.hillayer1 = Hilbert_Sa(4,1)
        self.hillayer2 = Hilbert_Sa(5,2)
        self.hillayer4 = Hilbert_Sa(6,4)
        self.hillayer8 = Hilbert_Sa(7,8)
        # ####################Smooth+eca layers######################
        self.smooth3_4 = nn.Sequential(
            nn.Conv2d(2048,2048,kernel_size=1,bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            eca_layer(2048)
        )

        ####################cat layers###################
    def _unsample_cat(self,x,y):
            _,_,h,w = y.shape
            x = F.interpolate(x,size=(h,w),mode='bilinear',align_corners=True)
            return torch.cat([x,y],dim=1)


    def forward(self,input):
        _,_,h,w = input.shape
        x0 = self.stem(input)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        # _,_,x3_h,x3_w = x3.shape
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(x5)
        x7 = self.stage7(x6)
        xhead = self.head(x7)

        # #############Lateral#################
        top1 = self.hillayer1(self.latlayer1(xhead))
        top2 = self.hillayer2(self.latlayer2(x5))
        top3 = self.hillayer4(self.latlayer3(x3))
        top4 = self.hillayer8(self.latlayer4(x2))


        # ##################cat layers##################
        top1_2 = self._unsample_cat(top1,top2)
        top2_3 = self._unsample_cat(top1_2,top3)
        top3_4 = self._unsample_cat(top2_3,top4)
        # #################smooth+eca##################
        x_eca = self.smooth3_4(top3_4)

        x = self.cov(x_eca)

        out = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel,gamma = 2,b=1):
        super(eca_layer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) /gamma))
        # t = int(abs(channel/4+b)/gamma)
        k = t if t%2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)

if __name__ == '__main__':
    model = efficient_v2(3,True)
    # model = Hilbert_Sa(4,1)
    a = torch.rand((1,3,512,512))
    b = model(a)
    print(b.shape)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.6fM" % (total / 1e6))
