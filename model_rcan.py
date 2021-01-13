import common

import torch.nn as nn
import torch
import numpy as np

import cv2
from skimage import io
import torch.nn.functional as F

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 64——>4
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 4——>64
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.LeakyReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        # modules_body.append(nn.BatchNorm2d(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class SRCNN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(SRCNN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        # scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # add deal graycode layer
        # self.models_graycode = commen.convert_graycode

        # define head module
        modules_head = [conv(8, n_feats, kernel_size)]  # (3,64,3 )
        # modules_head = [nn.Conv2d(8, 64, kernel_size=9, padding=4)]  # (3,64,3 )
        self.bn1 = nn.BatchNorm2d(64)
        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(nn.BatchNorm2d(n_feats))

        # define tail module
        modules_tail = [
            # common.Upsampler(conv, 2, n_feats, act=False),
            # conv(n_feats, args.n_colors, kernel_size)]
            conv(n_feats, 8, kernel_size)]

        self.head = nn.Sequential(*modules_head)

        self.relu1 = nn.ReLU(inplace=True)
        # self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        # self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, not_8):
        # print(inte.shape)
        inte = self.head(x)
        # print(x.shape)
        x = F.relu(self.bn1(inte))
        # x1 = x.cpu().detach().data.numpy()[0, 0, :, :]
        # np.save('/export/liuzhe/program2/SRCNN/results/x1.npy', x1)

        x = self.body(x)  # [8,64,48,48]
        # x = F.leaky_relu(x)
        # print(x.shape)
        # x2 = x.cpu().detach().data.numpy()[0, 0, :, :]
        # np.save('/export/liuzhe/program2/SRCNN/results/x2.npy', x2)

        x = self.tail(x + inte)      # [8,8,96,96]
        # print(x.shape)
        # x = F.leaky_relu(self.bn2(x)+inte)
        # x = self.conv4(x)

        # x3 = x.cpu().detach().data.numpy()[0, 0, :, :]
        # np.save('/export/liuzhe/program2/SRCNN/results/x3.npy', x3)
        # exit(-1)

        # x = self.add_mean(x)

        return x
