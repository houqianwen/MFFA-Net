import time
import numpy as np
import torch
import torch.nn.functional as F
from AMGNet import AMGNet
from ABMSDNet import ABMSDNet
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class DehazeNet(nn.Module): # Generator
    def __init__(self, in_features=6, channel=12):
        super(DehazeNet, self).__init__()
        self.AMGNet = AMGNet()    # Attention Map Generator Network
        self.ABMSDNet = ABMSDNet() # Attention-base Boosted Multiscale Dehazing Network
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1) # Output layer



    def forward(self, x):

        att_map1, att_map3, att_map5, attention_map = self.AMGNet(x)  #上分支蒸馏，单通道特征图
        feature_maps = self.ABMSDNet(x)  #下分支多尺度 3通道
        #res8_o1,res4_o1,res分别是第二层，第三层，最后一层
        res = self.conv_output(attention_map * feature_maps)  #res输出3通道数

        return att_map1, att_map3, att_map5, res


if __name__ == '__main__':
    net = DehazeNet().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(out[4].shape)
