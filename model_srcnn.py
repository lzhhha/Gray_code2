import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch
import cv2
# from skimage import io
import matplotlib.pyplot as plt

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.bn = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)


    def forward(self, x, not_8):
        # it = not_8.cuda().float()
        # print(x.shape)
        inte = x
        # x /= 255
        # print(torch.mean(x))
        out = self.conv1(x)     # [4,64,32,32]
        out = F.relu(self.bn1(out))
        # print(out)
        # print(torch.mean(out))     # max 2
        # exit(-1)

        out = self.conv2(out)     # 0.9
        out = F.relu(self.bn2(out))
        # print(out)
        # print(torch.mean(out))
        # exit(-1)
        out = self.conv3(out)  # [4,8,32,32]
        out = F.relu(self.bn3(out)+inte)
        out = self.conv4(out)

        # out += inte
        # out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = out.cuda()
        # it = F.upsample_bilinear(it, scale_factor=2)
        # out = F.upsample_bilinear(out, scale_factor=2)

        # out = self.conv4(out)
        # print(torch.mean(out))
        # exit(-1)

        # out = F.relu(out)

        # out += it
        # out = self.conv4(out)
        # out = F.sigmoid(out)
        # print(torch.mean(out))
        return out

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.008)
                # nn.init.kaiming_normal_(m.weight.data)
                # nn.init.kaiming_uniform_(m.weight.data)
                # nn.init.normal_(m.weight.data)  # normal: mean=0, std=1     正态分布初始化
                # nn.init.xavier_normal_(m.weight.data)
                # torch.nn.init.sparse_(m.weight.data)

                # nn.init.constant_(m.bias, 0)
