import torch.nn as nn
import torch
import torch.nn.functional as F


class DPyConv(nn.Module):
    def __init__(self, channel):
        super(DPyConv, self).__init__()
        self.channel = channel
        self.g_conv1 = nn.Conv2d(
            self.channel,
            self.channel // 2,
            (3, 3),
            dilation=1,
            groups=self.channel // 2,
            padding=1,
        )
        self.g_conv2 = nn.Conv2d(
            self.channel,
            self.channel // 2,
            (3, 3),
            dilation=2,
            groups=self.channel // 2,
            padding=2,
        )
        self.g_conv3 = nn.Conv2d(
            self.channel,
            self.channel // 2,
            (3, 3),
            dilation=3,
            groups=self.channel // 2,
            padding=3,
        )
        self.g_conv4 = nn.Conv2d(
            self.channel,
            self.channel // 2,
            (3, 3),
            dilation=4,
            groups=self.channel // 2,
            padding=4,
        )
        self.conv = nn.Conv2d(2 * self.channel, self.channel, (1, 1))

    def forward(self, x):
        x_1 = self.g_conv1(x)
        x_2 = self.g_conv2(x)
        x_3 = self.g_conv3(x)
        x_4 = self.g_conv4(x)
        x_ = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x_ = self.conv(x_)
        return x_
