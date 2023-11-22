import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1
torch.backends.cudnn.enabled = False

#Attention-based Feature Aggregation Module
class AFAM(nn.Module):
    def __init__(self,in_features=None):
        super(AFAM,self).__init__()

        self.conv1 = nn.Conv2d(in_features*2, in_features, 1, bias=False)
        self.conv2 = nn.Conv2d(in_features, 1, 3, 1, 1, bias=False)
        self.Th = nn.Sigmoid()


    def forward(self, x, y):

        res = torch.cat([x, y], dim=1)
        x1 = self.conv1(res)
        x2 = self.conv2(x1)
        x2 = self.Th(x2)
        out = x2 * x1

        return out
#####################################################


def make_model(args, parent=False):
    return ABMSDNet()

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)  #沿着通道维度进行拼接
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

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


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out

#############编码器的残差组以及用于编解码之间的残差
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.IN = nn.InstanceNorm2d(channels)  #######
        self.relu = nn.PReLU()
        self.conv3 = ConvLayer(channels, 1, kernel_size=3, stride=1)
        self.Th = nn.Sigmoid()       ##########

    def forward(self, x):
        residual = x
        out = self.relu(self.IN(self.conv1(x)))
        out = out + residual
        out = self.conv2(out)
        map = self.Th(self.conv3(out))
        out = out * map + residual
        return out

class ResidualBlock_ori(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_ori, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ABMSDNet(nn.Module):
    def __init__(self, res_blocks=18):
        super(ABMSDNet, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode='iter2')

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))


        self.afam4=AFAM(128)
        self.afam3=AFAM(64)
        self.afam2=AFAM(32)
        self.afam1=AFAM(16)


        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x):
        res1x = self.conv_input(x) #输入3通道输出16通道
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1) #通道数减半8
        feature_mem = [res1x_1] #通道数为8
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x) #输入16通道输出32通道
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1) #通道数减半16
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        res_dehaze = res16x
        in_ft = res16x*2

        res16x = self.dehaze(in_ft) + in_ft - res_dehaze #最顶层的18个残差组
######################################### Decoder ##################################################

        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)#通道数减半
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear') #上采样到和res8x一样的大小
        # res8x = torch.add(res16x, res8x)
        # res8x = self.dense_4(res8x) + res8x - res16x

######################### Attention-base SOS Module #########################
        res8x = self.afam4(res16x, res8x) + res16x  ####先编解码特征融合（res8x编码器特征，res16x上采样后的解码器特征），融合后的特征再加上res16x
        res8x = self.dense_4(res8x) + res16x ##将得到的res8x特征输入到密集块中，再和rex16x相加得到输出
######################### End #########################

        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.convd8x(res8x)

        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        # res4x = torch.add(res8x, res4x)
        # res4x = self.dense_3(res4x) + res4x - res8x

        res4x = self.afam3(res8x, res4x) + res8x      #########
        res4x = self.dense_3(res4x) +  res8x

        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.convd4x(res4x)

        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        # res2x = torch.add(res4x, res2x)
        # res2x = self.dense_2(res2x) + res2x - res4x

        res2x = self.afam2(res4x, res2x) + res4x  ######
        res2x = self.dense_2(res2x) + res4x

        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')

        # x = torch.add(res2x, x)
        # x = self.dense_1(x) + x - res2x

        x = self.afam1(res2x, x) + res2x #########
        x = self.dense_1(x) + res2x

        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        # x = self.conv_output(x)
        # return x
        return x           ###########多尺度

if __name__ == '__main__':
    net = ABMSDNet().cuda()
    input_tensor = torch.Tensor(np.random.random((1,3,256,256))).cuda()
    start = time.time()
    out = net(input_tensor)
    print(out[0].shape) #第二层，第三层分别是64，128
    print(out[1].shape)
    print(out[2].shape)
    # print(out[3].shape)
    end = time.time()
    print('Process Time: %f'% (end-start))
    print(out)