import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class RESBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RESBlock, self).__init__()

        self.split_conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(15, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x1_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.split_conv_x2_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(13, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x2_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 13)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.split_conv_x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(11, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 11)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.split_conv_x4_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x4_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.sum_conv_x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.sum_conv_x2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.sum_conv_x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        init = x

        split_conv_x1 = self.split_conv_x1_1(x)
        split_conv_x1 = self.split_conv_x1_2(split_conv_x1)
        
        split_conv_x2 = self.split_conv_x2_1(x)
        split_conv_x2 = self.split_conv_x2_2(split_conv_x2)
        
        split_conv_x3 = self.split_conv_x3_1(x)
        split_conv_x3 = self.split_conv_x3_2(split_conv_x3)
        
        split_conv_x4 = self.split_conv_x4_1(x)
        split_conv_x4 = self.split_conv_x4_2(split_conv_x4)
        
        x = torch.cat([init, split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4], dim=1)
        
        x = self.sum_conv_x1(x)
        x = self.sum_conv_x2(x)
        x = self.sum_conv_x3(x)

        return x
        

class WCBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(WCBlock, self).__init__()

        self.split_conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(15, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x1_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.split_conv_x2_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.split_conv_x2_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(15, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):

        split_conv_x1 = self.split_conv_x1_1(x)
        split_conv_x1 = self.split_conv_x1_2(split_conv_x1)
        
        split_conv_x2 = self.split_conv_x2_1(x)
        split_conv_x2 = self.split_conv_x2_2(split_conv_x2)
        
        x = torch.cat([split_conv_x1, split_conv_x2], dim=1)
        x = self.batch_norm(x)
        x = nn.ReLu(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_blk = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_blk(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_blk = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_blk(x)
        return x

class BUNet(nn.Module):
    def __init__(self, n_classes):
        super(BUNet, self).__init__()
        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bridge = ConvBlock(512, 1024)

        self.res_block = RESBlock(1024, 1024)
        self.wc_block = WCBlock(512, 512)

        self.decoder1 = DecoderBlock(1024 + 512, 512)
        self.decoder2 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder4 = DecoderBlock(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b = self.bridge(p4)

        wc = self.wc_block(s4)

        res = self.res_block(b)

        d1 = self.decoder1(res, wc)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        outputs = self.out_conv(d4)
        outputs = torch.sigmoid(outputs)
        return outputs

# 모델 초기화
model = BUNet(n_classes=21)
print(model)