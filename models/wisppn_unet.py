import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack as DCN

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, deformable=False):
        super(DoubleConv, self).__init__()
        self.deform_groups = 3
        self.deformable = deformable

        if self.deformable:
            assert in_channels % self.deform_groups == 0, "in_channels must be divisible by deform_groups"
            assert out_channels % self.deform_groups == 0, "out_channels must be divisible by deform_groups"
        
        self.double_conv = nn.Sequential(
            DCN(in_channels, out_channels, kernel_size=3, padding=1, deform_groups=self.deform_groups) if self.deformable else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DCN(out_channels, out_channels, kernel_size=3, padding=1, deform_groups=self.deform_groups) if self.deformable else nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, deformable=False):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, deformable=deformable)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, deformable=False)  # 不使用DCN

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64, deformable=False)  # 使用普通卷积层
        self.down1 = Down(64, 128, deformable=True)  # 使用DCN
        self.down2 = Down(128, 256, deformable=True)  # 使用DCN
        self.down3 = Down(256, 512, deformable=True)  # 使用DCN
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, deformable=True)  # 使用DCN
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)  # 使用普通卷积层

        self.adaptive_pool = nn.AdaptiveAvgPool2d((17, 17))  # 新增的自适应池化层

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        x = self.adaptive_pool(logits)  # 新增的自适应池化层

        return x
