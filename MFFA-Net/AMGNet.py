import torch
import torch.nn as nn
import torch.nn.functional as F
from base_networks import FastDeconv
import numpy as np
import time
torch.backends.cudnn.benchmark = False
#################注意力分支(上分支)，知识蒸馏
#Attention-based Feature Aggregation Module 基于注意力的特征融合模块
class AFAM(nn.Module):
    def __init__(self,in_features=32):
        super(AFAM,self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features//2, 1, bias=False)
        self.conv2 =nn.Conv2d(in_features//2, 1, 3, 1, 1, bias=False)
        self.Th = nn.Sigmoid()


    def forward(self, x, y):

        res = torch.cat([x, y], dim=1)
        x1 = self.conv1(res)
        x2 = self.conv2(x1)
        x2 = self.Th(x2)
        out= x2 *  x1

        return out
############################################

# class AFAM(nn.Module):
#     def __init__(self, in_features=16):
#         self.init__ = super(AFAM, self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(in_features, in_features,kernel_size=3,stride=1,padding=1)
#         self.conv2_1 = nn.Conv2d(in_features, in_features //4,kernel_size=3,stride=1,padding=1)
#         self.conv2_2 = nn.Conv2d(in_features,in_features //4,kernel_size=3, stride=1,padding=1)
#         self.conv3 = nn.Conv2d(in_features //4,in_features,kernel_size=3,stride=1,padding=1)
#         self.Th = nn.Sigmoid()
#
#     def forward(self,x1,x2):
#         weight = self.Th(self.conv1(x1+x2))
#         wresid_1 = x1 + x1.mul(weight)
#         wresid_2 = x2 + x2.mul(weight)
#
#         x1_2 = self.conv2_1(wresid_1)
#         x2_2 = self.conv2_2(wresid_2)
#
#         out = self.relu(self.conv3(x1_2+x2_2))
#
#         return out
###############注意力特征融合
# class AFAM(nn.Module):
#     '''
#     多特征融合 AFF
#     '''
#
#     def __init__(self, channels=16,r=4):
#         self.init__ = super(AFAM, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, y):
#         xa = x + y
#         xl = self.local_att(xa)
#         # print(xl)
#         xg = self.global_att(xa)
#         # print(xg)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#
#         xo = 2 * x * wei + 2 * y * (1 - wei)
#         return xo
##################################################

#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, nc,number, norm_layer = nn.BatchNorm2d):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
                                   norm_layer(nc),
                                   nn.PReLU(nc),
                                   nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
                                   norm_layer(nc))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #最大池化
        #全连接层
        self.conv2 = nn.Sequential(nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False),
                                    nn.Sigmoid())
    def forward(self, x):

        out1 = self.conv2(self.avg_pool(self.conv1(x)))
        out2 = self.conv2(self.max_pool(self.conv1(x)))

        return out1, out2    #分别代表输入特征图在平均池化和最大池化操作下的注意力特征图（单通道）

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, nc, number, norm_layer = nn.BatchNorm2d):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias=False),
                                   norm_layer(nc),
                                   nn.PReLU(nc),
                                   nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False),
                                   norm_layer(number))

        self.conv2 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)

        self.conv5 = nn.Sequential(nn.Conv2d(number*4,1,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False),
                                   nn.Sigmoid())

    def forward(self, x):

        x = self.conv1(x)
        x1 = x
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        se = torch.cat([x1, x2, x3, x4], dim=1)
        
        out = self.conv5(se)
        
        return out

# Triple Feature Attention Module 三特征注意模块
class TFAM(nn.Module):
    def __init__(self,nc, number = 4, norm_layer = nn.BatchNorm2d):
        super(TFAM,self).__init__()
        self.CA = ChannelAttention(nc,number)  #通道注意力C×1×1
        self.MSSA = SpatialAttention(nc,number) #空间注意力1×W×H
        self.AFAM = AFAM()    #基于注意力的特征融合模块
    def forward(self,x):

        ca_1,ca_2 = self.CA(x)

        ca_map1 = ca_1 * x
        ca_map2 = ca_2 * x

        mssa_map1 = self.MSSA(ca_map1)*ca_map1
        mssa_map2 = self.MSSA(ca_map2)*ca_map2

        # afam = self.AFAM(mssa_map1,mssa_map2)
        # out= afam + x
        out = self.AFAM(mssa_map1, mssa_map2)

        return out

# Attention Map Generator
class AMGNet(nn.Module):
    def __init__(self, input_nc=3, in_features=32, n_residual_att=6):
        super(AMGNet, self).__init__()
        # Preprocess
        self.deconv = FastDeconv(input_nc, input_nc, 3, padding=1)

	    # Attention
        att = [ nn.ReflectionPad2d(3),
	            nn.Conv2d(input_nc, in_features//2, 7),
                nn.BatchNorm2d(in_features//2),
                nn.PReLU()]
        # for _ in range(n_residual_att):
        #     att += [TFAM(in_features//2)]
        self.att1=TFAM(in_features // 2)
        self.att2= TFAM(in_features // 2)
        self.att3=TFAM(in_features // 2)
        self.att4=TFAM(in_features // 2)
        self.att5=TFAM(in_features // 2)
        self.att6=TFAM(in_features // 2)
        # out
        att7= [nn.ReflectionPad2d(3),
                nn.Conv2d(in_features//2, 1, 7),
                nn.Sigmoid()]
    
        self.att = nn.Sequential(*att)
        self.att7 = nn.Sequential(*att7)

#####################
    def forward(self, x):
        x_deconv = self.deconv(x)
        # attention_map = self.att(x_deconv)
        # maps = [] #存储中间层特征
        # for module in self.att[:-1]:
        #     x = module(x_deconv)
        #     maps.append(x)
        att_map = self.att(x_deconv)
        # print(att_map.shape)
        att_map1 = self.att1(att_map)
        # print(att_map1.shape)
        att_map2 = self.att2(att_map1)
        att_map3 = self.att3(att_map2)
        att_map4 = self.att4(att_map3)
        att_map5 = self.att5(att_map4)
        att_map6 = self.att6(att_map5)
        attention_map = self.att7(att_map6)
        return att_map1,att_map3,att_map5, attention_map

        
if __name__ == '__main__':
    net = AMGNet().cuda()
    # net.eval()
    input_tensor = torch.Tensor(np.random.random((1,3,256,256))).cuda()
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(out)


