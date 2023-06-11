from test.Hilbert import Tra_Hilbert
import torch
import torch.nn.functional as F
from torch import nn


'''基于希尔伯特曲线遍历的空间子注意力模块'''
class Hilbert_Sa(nn.Module):
    '''order为该模块所采用的哪一阶希尔伯特曲线'''
    def __init__(self,order,pool_size):
        super(Hilbert_Sa, self).__init__()
        #降通道->B*1*H*W
        # self.dc = nn.Conv2d(in_channels,1,1)
        # self.relu = nn.ReLU(inplace=True)
        # self.cov2d = nn.Conv2d(2,1,1,1)
        #转化为Q,K,V

        # self.trans_q = nn.Conv2d(2,1,1)
        # self.trans_k = nn.Conv2d(2,1,1)
        # self.trans_v = nn.Conv2d(2,1,1)

        self.softmax = nn.Softmax(1)
        self.sigmiod = nn.Sigmoid()
        self.order = order
        self.pool_size = pool_size
        self.path_num = int(((2**self.order)/self.pool_size)**2)
        self.channel = 2*(self.pool_size**2)
        self.trans_q = nn.Conv2d(self.channel,self.channel,1)
        self.trans_k = nn.Conv2d(self.channel,self.channel,1)
        self.trans_v = nn.Conv2d(self.channel,self.channel,1)
        #
        # self.pool_size = pool_size
        #使用Conv1d转化为QKV
        # self.trans_q = nn.Conv1d((2**self.order)**2,(2**self.order)**2,2)
        # self.trans_k = nn.Conv1d((2**self.order)**2,(2**self.order)**2,2)
        # self.trans_v = nn.Conv1d((2**self.order)**2,(2**self.order)**2,2)

        # self.avg_pool = nn.AdaptiveAvgPool2d(self.pool_size)
        # self.up = nn.Upsample(scale_factor=16/self.pool_size,mode='bilinear',align_corners=True)

        # self.eca_cov = eca_cov

        self.pre_a = nn.Conv2d(2,1,1)

    def forward(self,x):
        r = x
        # eca = self.eca_cov(x)
        b,c,h,w = x.shape
        hil,rehil = Tra_Hilbert(self.order)
        mean = torch.mean(x,dim=1,keepdim=True)
        max,_ = torch.max(x,dim=1,keepdim=True)
        x1 = torch.cat([mean,max],1)#B*2*H*W
        # x1 = torch.sum(x1,dim=1)     #B*1*H*W
        x2 =  x1.view(b,2,1,-1)    #B*2*1*HW
        d = torch.chunk(x2,h*w,3)
        ########希尔伯特曲线遍历过程###############
        l = []
        for i in range((2**self.order)*(2**self.order)):
            num = hil[i]
            l.append(d[num])
        x2 = torch.cat(l,3) #B*2*1*HW
        m = torch.chunk(x2,self.path_num,3)#B*2*1*ps^2
        x2 = torch.cat(m,dim=2).permute(0,1,3,2).contiguous()   #B*2*ps^2*p_n
        x2 = x2.view(b,-1,1,self.path_num).contiguous()

        ##################二维卷积#########################
        Q = self.trans_q(x2) #B*c*1*p_n
        Q = torch.squeeze(Q,2) #B,c,p_n
        K = self.trans_k(x2)
        K = torch.squeeze(K,2)
        V = self.trans_v(x2)
        V = torch.squeeze(V,2)
        Q = Q.permute(0,2,1).contiguous() #B*p_n*c
        V = V.permute(0,2,1).contiguous()
        #################一维卷积###########################
        # x2 = torch.squeeze(x2,2).permute(0,2,1).contiguous() #B,HW,2
        # Q = self.trans_q(x2) #B,HW,1
        # K = self.trans_k(x2).permute(0,2,1).contiguous() #B,1,HW
        # V = self.trans_v(x2) #B,HW,1

        #################################################

        a = self.softmax(torch.matmul(Q,K)) #B*p_n*p_n
        x3 = torch.matmul(a,V)    #B*p_n*c
        x3 = x3.permute(0,2,1).contiguous() #B*c*p_n
        x3 = x3.unsqueeze(2)      #B*c*1*p_n

        x3 = x3.view(b,2,-1,self.path_num).permute(0,1,3,2).contiguous() #b*2*p_n*ps^2
        m = torch.chunk(x3,self.path_num,dim=2)  #b*2*1*ps^2
        x3 = torch.cat(m,dim=3)
        # print(x3.shape)
        d = torch.chunk(x3,h*w,dim=3)
        ######反希尔伯特曲线遍历返回原像素点位置####
        l1 = []
        for i in range((2**self.order)*(2**self.order)):
            num = rehil[i]
            l1.append(d[num])
        x3 = torch.cat(l1,3)
        ############################################
        b = x3.view(b,2,h,-1) #B*1*H*W
        b = self.pre_a(b)
        # b = b.sum(1) #B*H*W
        # b = b.unsqueeze(1) #B*1*H*W
        # b = b*eca
        # b = self.up(b)
        out = self.sigmiod(b)*r
        return out

class SE(nn.Module):
    def __init__(self,in_channel):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel = in_channel
        # self.fc1 = nn.Conv2d(in_planes,in_planes//16,1,bias=False)
        # self.rule1 = nn.ReLU()
        # self.fc1 = nn.Conv2d(in_planes,in_planes//16,1,bias=False)

        self.trans_q = nn.Conv2d(self.channel ,self.channel ,1)
        self.trans_k = nn.Conv2d(self.channel ,self.channel ,1)
        self.trans_v = nn.Conv2d(self.channel ,self.channel ,1)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        r = x

        ###########################################
        avg_x = self.avg_pool(x)   #B*C*1*1
        max_x = self.max_pool(x)   #B*C*1*1
        x1 = torch.cat([avg_x,max_x],dim=2) #B*C*2*1
        x2 = torch.sum(x1,dim=2)   #B*C*1
        x2 = x2.unsqueeze(2) #B*C*1*1

        ###########################################

        # avg_x = self.avg_pool(x)
        # avg_x = avg_x.permute(0,2,3,1).contiguous()
        x2_q = self.trans_q(x2).permute(0,2,1,3).contiguous() #B*1*C*1
        x2_k = self.trans_k(x2).permute(0,2,3,1).contiguous() #B*1*1*C
        x2_v = self.trans_v(x2).permute(0,2,1,3).contiguous() #B*1*C*1
        x3 = torch.matmul(self.softmax(torch.matmul(x2_q,x2_k)),x2_v).permute(0,2,1,3).contiguous()


        # avg_q = avg_q.permute(0,1,3,2).contiguous()
        # # avg_k = avg_k.permute(0,2,3,1).contiguous()
        # avg_v = avg_v.permute(0,1,3,2).contiguous()
        # avg_out = torch.matmul(self.softmax(torch.matmul(avg_q,avg_k)),avg_v)
        # avg_out = avg_out.permute(0,2,1,3).contiguous()
        #
        # max_x = self.max_pool(x)
        # max_x = max_x.permute(0, 2, 3, 1).contiguous()
        # max_q = self.trans_q(max_x)
        # max_k = self.trans_k(max_x)
        # max_v = self.trans_v(max_x)
        # max_q = max_q.permute(0,1,3,2).contiguous()
        # # max_k = max_k.permute(0,2,3,1).contiguous()
        # max_v = max_v.permute(0,1,3,2).contiguous()
        # max_out = torch.matmul(self.softmax(torch.matmul(max_q,max_k)),max_v)
        # max_out = max_out.permute(0,2,1,3).contiguous()

        # out = avg_out+max_out
        out = r*self.sigmoid(x3)
        return out


class eca_cov(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_cov, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


if __name__ == '__main__':
    a = torch.rand((1,512,16,16))
    # a = torch.mean(s,1)
    # b,_ = torch.max(s,1)
    # print(b.shape)
    # c = torch.cat([a,b],1)
    # print(c.shape)
    # conv_1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=2)
    # print(conv_1(a).shape)
    # conv = nn.Conv2d(1,1,1)
    # b  = conv(a)
    # print(b.shape)
    # b = torch.sum(a,2)
    # print(b.shape)
    # a = a.view(1,2,1,-1)
    # b = torch.chunk(a,64,3)
    # print(b[1].shape)
    # d = torch.chunk(a,64,3)
    # print(d[1].shape)
    # sa = Hilbert_Sa(3)
    # b = sa(a)
    # print(b.shape)
    # model = SE(512)
    # b = model(a)
    # print(b.shape)
    # model = PSPNet(num_classes=3,downsample_factor=16,pretrained=True,aux_branch=False)
    model = Hilbert_Sa(4,1)
    b = model(a)
    print(b.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.6fM" % (4*(total/1e6)))


