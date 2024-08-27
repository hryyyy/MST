from Xception import efficient
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, einsum
import torchvision
import numpy as np
import torch.nn.functional as F
from myutils.utils import SRMConv2d_simple, SRMConv2d_Separate,learn_constrain_simple,learn_SRMConv2d_simple
import math
from timm.models.layers import DropPath,trunc_normal_
from PIL import Image

class MSCA(nn.Module):
    def __init__(self, in_channels, ratio):
        super(MSCA, self).__init__()
        self.max_pool_3x3 = nn.AvgPool2d(3)
        self.max_pool_5x5 = nn.AvgPool2d(5)
        self.max_pool_7x7 = nn.AvgPool2d(7)
        self.gobal_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gobal_max_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv2d(3,1,1,bias=False)
        self.linear1 = nn.Linear(3, 1, bias=False)
        self.linear2 = nn.Linear(3, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        m3 = self.max_pool_3x3(x)
        m5 = self.max_pool_5x5(x)
        m7 = self.max_pool_7x7(x)

        p1, p2, p3 = self.gobal_avg_pool(m3), self.gobal_avg_pool(m5), self.gobal_avg_pool(m7)
        p4, p5, p6 = self.gobal_max_pool(m3), self.gobal_max_pool(m5), self.gobal_max_pool(m7)

        a_p = torch.cat([p1, p2, p3], dim=2).transpose(2, 3)
        a_p = self.sharedMLP(self.linear1(a_p).transpose(2, 3))

        m_p = torch.cat([p4, p5, p6], dim=2).transpose(2, 3)
        m_p = self.sharedMLP(self.linear2(m_p).transpose(2, 3))

        # p = self.linear(p).transpose(2,3)
        # x = x * p + x
        return self.sigmoid(a_p + m_p)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MBConv(nn.Module):

    def __init__(self,in_channel,squeeze):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel*squeeze,1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel*squeeze)

        self.dwconv = nn.Conv2d(in_channel*squeeze,in_channel*squeeze,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel*squeeze)

        self._se_reduce = nn.Conv2d(in_channel*squeeze,in_channel,1)
        self._se_expand = nn.Conv2d(in_channel,in_channel*squeeze,1)

        self.conv2 = nn.Conv2d(in_channel*squeeze,in_channel,1,bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)

        self.drop = nn.Dropout(0.4)
        self._swish = MemoryEfficientSwish()

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self._swish(x)

        x = self.dwconv(x)
        x = self.bn2(x)
        x = self._swish(x)

        x_squeeze = F.adaptive_avg_pool2d(x,1)
        x_squeeze = self._se_reduce(x_squeeze)
        x_squeeze = self._swish(x_squeeze)
        x_squeeze = self._se_expand(x_squeeze)
        x = torch.sigmoid(x_squeeze) * x

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.drop(x)

        x = x + input
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class Dense(nn.Module):
    def __init__(self,in_channel):
        super(Dense, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, in_channel, 5, padding=2)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel * 2, in_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * in_channel)
        self.conv3 = nn.Conv2d(in_channel * 3, in_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3 * in_channel)
        self.conv_last = nn.Conv2d(in_channel * 4, in_channel, 1)
        self.bn4 = nn.BatchNorm2d(4 * in_channel)
        self.bn_last = nn.BatchNorm2d(in_channel)

    def forward(self,x):
        feature_maps0 = self.conv0(x)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0, feature_maps1],dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_, feature_maps2],dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_, feature_maps3],dim=1)
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))),inplace=True)
        return feature_maps

class Texture(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_extract = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        # self.conv_extract = MBConv(in_channel,1)
        self.dense = Dense(in_channel)

    def forward(self, feature_maps, size=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(size) == tuple:
            attention_size = (int(H * size[0]), int(W * size[1]))
        else:
            attention_size = (size.shape[2], size.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)

        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        feature_maps = self.dense(feature_maps)
        return feature_maps

class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        # tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0)
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        # tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0)
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class CDC_block1_1(nn.Module):
    def __init__(self,in_channel,out_channel,ba_conv1=Conv2d_Hori_Veri_Cross,ba_conv2=Conv2d_Diag_Cross):
        super(CDC_block1_1, self).__init__()
        self.conv1 = nn.Sequential(
            ba_conv1(in_channel,out_channel,kernel_size=3,stride=2,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True)
        )
        self.conv2 = nn.Sequential(
            ba_conv2(in_channel,out_channel,kernel_size=3,stride=2,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True)
        )
        self.mbconv = MBConv(out_channel,6)
        self.sge = SpatialGroupEnhance(8)
        self.act = nn.Hardswish(True)

    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.mbconv(out)
        out = self.act(self.sge(out) + out)
        return out

class CDC_block1_2(nn.Module):
    def __init__(self,in_channel,out_channel,ba_conv1=Conv2d_Hori_Veri_Cross,ba_conv2=Conv2d_Diag_Cross):
        super(CDC_block1_2, self).__init__()
        self.conv1 = nn.Sequential(
            ba_conv1(in_channel,in_channel,kernel_size=3,stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv1(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            ba_conv2(in_channel, in_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv2(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.mbconv = MBConv(out_channel,6)
        self.sge = SpatialGroupEnhance(8)
        self.act = nn.Hardswish(True)

    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.mbconv(out)
        out = self.act(self.sge(out) + out)
        return out

class CDC_block1_3(nn.Module):
    def __init__(self,in_channel,out_channel,ba_conv1=Conv2d_Hori_Veri_Cross,ba_conv2=Conv2d_Diag_Cross):
        super(CDC_block1_3, self).__init__()
        self.conv1 = nn.Sequential(
            ba_conv1(in_channel,in_channel,kernel_size=3,stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv1(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            ba_conv2(in_channel, in_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv2(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.mbconv = MBConv(out_channel,6)
        self.sge = SpatialGroupEnhance(8)
        self.act = nn.Hardswish(True)

    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.mbconv(out)
        out = self.act(self.sge(out) + out)
        return out

class CDC_block1_4(nn.Module):
    def __init__(self,in_channel,out_channel,ba_conv1=Conv2d_Hori_Veri_Cross,ba_conv2=Conv2d_Diag_Cross):
        super(CDC_block1_4, self).__init__()
        self.conv1 = nn.Sequential(
            ba_conv1(in_channel,in_channel,kernel_size=3,stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv1(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            ba_conv2(in_channel, in_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(True),
            ba_conv2(in_channel, out_channel, kernel_size=3, stride=1,theta=0.8),
            nn.BatchNorm2d(out_channel),
            nn.Hardswish(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.mbconv = MBConv(out_channel,6)
        self.sge = SpatialGroupEnhance(8)
        self.act = nn.Hardswish(True)

    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.mbconv(out)
        out = self.act(self.sge(out) + out)
        return out

class AMFB(nn.Module):
    def __init__(self,in_channel):
        super(AMFB, self).__init__()
        self.ch_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel,in_channel,1,bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.ch_att(x) * x
        return x

class FPFM(nn.Module):
    def __init__(self):
        super(FPFM, self).__init__()
        self.text_conv1 = nn.Conv2d(48,24,1)
        self.text_conv2 = nn.Conv2d(24,32,1)
        self.text_conv3 = nn.Conv2d(32,56,1)
        self.text_conv4 = nn.Conv2d(56,160,1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(24,32,3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,56,3,stride=2,padding=1),
            nn.BatchNorm2d(56),
            nn.Hardswish(True),
            nn.Conv2d(56,160,3,stride=2,padding=1),
            nn.BatchNorm2d(160),
            nn.Hardswish(True)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.amfb1 = AMFB(32)
        self.amfb2 = AMFB(160)

        self.layar = nn.Sequential(
            nn.Conv2d(160,160,3,1,1),
            nn.BatchNorm2d(160),
            nn.Hardswish(True)
        )
        # self.layar = nn.Conv2d(160,160,1)
        self.msca = MSCA(160,8)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self,text1,text2,high):
        text = self.text_conv1(text1) + text2 #24 112
        text_branch = self.text_conv2(self.maxpool(text)) # 32 56
        text = self.conv1(text) + self.amfb1(text_branch) #32 56

        text_branch = self.text_conv3(self.maxpool(text_branch)) # 56 28
        text_branch = self.text_conv4(self.maxpool(text_branch)) # 160 14
        text = self.conv2(text) + self.amfb2(text_branch) # 160 14

        # out = self.fu(high,text)
        out = self.layar(high+text)
        out = self.msca(out) * out + out
        out = self.adapool(out)
        out = out.view(out.size(0),-1)
        return out

class AFFM(nn.Module):
    def __init__(self,in_channel):
        super(AFFM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottconv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel//8,1,bias=False),
            nn.BatchNorm2d(in_channel//8),
            nn.ReLU(),
            nn.Conv2d(in_channel//8,in_channel,1,bias=False),
            nn.BatchNorm2d(in_channel)
        )

    def forward(self,rgb,text):
        fea = torch.cat([rgb,text],dim=1)
        att = self.bottconv(self.gap(fea)).squeeze()
        att_rgb,att_text = torch.chunk(att,2,1)
        combined = torch.stack((att_rgb,att_text),dim=1)
        combined = F.softmax(combined,dim=1)
        att_rgb,att_text = combined[:,0,:],combined[:,1,:]

        rgb = rgb + rgb * att_rgb
        text = text + text * att_text
        return rgb,text

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rgb_cnn = efficient.EfficientNet2.from_pretrained('efficientnet-b4', num_classes=2)
        self.cons = learn_constrain_simple(3)

        self.text_block1 = Texture(48)
        self.text_block2 = Texture(24)

        self.cd1_1 = CDC_block1_1(3,24)
        self.cd1_2 = CDC_block1_2(24,32)
        self.cd1_3 = CDC_block1_3(32,56)
        self.cd1_4 = CDC_block1_4(56,160)

        self.msca = MSCA(160,8)
        self.fpfm = FPFM()
        #self.affm = AFFM(320)

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1952,2)
        )

    def forward(self, x):
        high = self.cons(x)

        rgb = self.rgb_cnn.start_fea(x)
        text_fea1 = self.text_block1(rgb,(0.25,0.25))

        rgb = self.rgb_cnn.extra_fea_part_1(rgb)
        text_fea2 = self.text_block2(rgb,(0.25,0.25))
        high = self.cd1_1(high)

        rgb = self.rgb_cnn.extra_fea_part_2(rgb)
        high = self.cd1_2(high)

        rgb = self.rgb_cnn.extra_fea_part_3(rgb)
        high = self.cd1_3(high)

        rgb = self.rgb_cnn.extra_fea_part_4(rgb)
        rgb = self.msca(rgb) * rgb + rgb
        high = self.cd1_4(high)
        hbp = self.fpfm(text_fea1,text_fea2,high)
        #rgb, hbp = self.affm(rgb, hbp) # Consider that the final output of the FPFM module does not use global average pooling

        rgb = self.rgb_cnn.extra_fea_part_5(rgb)
        rgb = F.adaptive_avg_pool2d(rgb,(1,1))
        rgb = rgb.view(rgb.size(0),-1)

        fea = torch.cat([hbp,rgb],dim=1)
        out = self.linear(fea)
        return out
