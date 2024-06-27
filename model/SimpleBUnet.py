import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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

class WC_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(WC_Block, self).__init__()

        self.split_conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(15, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 15)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x2_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x2_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(15, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.conv_sum = nn.Conv2d(2* out_channels, out_channels, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        split_conv_x1 = self.split_conv_x1_1(x)
        split_conv_x1 = self.split_conv_x1_2(split_conv_x1)
        split_conv_x2 = self.split_conv_x2_1(x)
        split_conv_x2 = self.split_conv_x2_2(split_conv_x2)
        x = torch.cat([split_conv_x1, split_conv_x2],dim=1)
        x = self.conv_sum(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class RES_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RES_Block, self).__init__()

        self.split_conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(15, 1), padding=(7, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.split_conv_x4_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x4_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 9), padding=(0, 4)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.sum_conv_x1 = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

        self.sum_conv_x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    def forward(self, x):
        init = x
        
        split_conv_x1 = self.split_conv_x1_1(x)
        split_conv_x1 = self.split_conv_x1_2(split_conv_x1)
        #init = F.interpolate(init, size= split_conv_x1.shape[2:4])

        split_conv_x4 = self.split_conv_x4_1(x)
        split_conv_x4 = self.split_conv_x4_2(split_conv_x4)

        x = torch.cat([init, split_conv_x1, split_conv_x4], dim=1)
        
        x = self.sum_conv_x1(x)
        x = self.sum_conv_x3(x)

        return x


class SimpleBUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleBUnet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.WC = WC_Block(512, 1024)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.RES3 = RES_Block(256,256)

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

        WC_block = self.WC(pool4)
        bottleneck = self.bottleneck(WC_block)

        upconv4 = self.upconv4(bottleneck)
        upconv4 = F.interpolate(upconv4, size=conv4.shape[2:4])

        cat4 = torch.cat((upconv4, conv4), dim=1)
        iconv4 = self.iconv4(cat4)
        upconv3 = self.upconv3(iconv4)

        RES_block = self.RES3(conv3)
        RES_block = F.interpolate(RES_block, size=upconv3.shape[2:4])
        cat3 = torch.cat((upconv3, RES_block), dim=1)
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