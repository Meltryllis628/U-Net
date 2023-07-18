import torch
import torch.nn as nn
from modules.Conv import ConvBlock, DeConvConnectBlock, DeConvBlock
from modules.DPyConv import DPyConv


class someNet(nn.Module):
    def __init__(self, channels=64) -> None:
        super(someNet, self).__init__()
        self.conv1 = ConvBlock(1, 1 * channels, 3, 2, 1)
        self.pyconv4 = DPyConv(1 * channels)
        self.conv2 = ConvBlock(1 * channels, 2 * channels, 3, 2, 1)
        self.pyconv3 = DPyConv(2 * channels)
        self.conv3 = ConvBlock(2 * channels, 4 * channels, 3, 2, 1)
        self.pyconv2 = DPyConv(4 * channels)
        self.conv4 = ConvBlock(4 * channels, 8 * channels, 3, 2, 1)
        self.pyconv1 = DPyConv(8 * channels)
        self.conv5 = ConvBlock(8 * channels, 16 * channels, 3, 2, 1)
        self.deconv1 = DeConvConnectBlock(8 * channels)
        self.deconv2 = DeConvConnectBlock(4 * channels)
        self.deconv3 = DeConvConnectBlock(2 * channels)
        self.deconv4 = DeConvConnectBlock(1 * channels)
        self.deconv = DeConvBlock(1 * channels, 1, 1, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x4 = self.pyconv1(x4)
        x = self.deconv1(x5,x4)
        x3 = self.pyconv2(x3)
        x = self.deconv2(x,x3)
        x2 = self.pyconv3(x2)
        x = self.deconv3(x,x2)
        x1 = self.pyconv4(x1)
        x = self.deconv4(x,x1)
        x = self.deconv(x)
        return x