import torch
import torch.nn as nn
import numpy as np
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.iconv4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.iconv3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.iconv2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.iconv1 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        bottleneck = self.bottleneck(pool4)

        upconv4 = self.upconv4(bottleneck)
        cat4 = torch.cat((upconv4, conv4), dim=1)
        iconv4 = self.iconv4(cat4)
        upconv3 = self.upconv3(iconv4)
        cat3 = torch.cat((upconv3, conv3), dim=1)
        iconv3 = self.iconv3(cat3)
        upconv2 = self.upconv2(iconv3)
        cat2 = torch.cat((upconv2, conv2), dim=1)
        iconv2 = self.iconv2(cat2)
        upconv1 = self.upconv1(iconv2)
        cat1 = torch.cat((upconv1, conv1), dim=1)
        iconv1 = self.iconv1(cat1)

        out = self.outconv(iconv1)
        out = self.softmax(out)
        return out