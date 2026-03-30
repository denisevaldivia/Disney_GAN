import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# Basic conv block
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with reflection padding"""
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=0, bias=False)
    )


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(inplace=True),

            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


def add_resblocks(channel_num, nr_blocks):
    return nn.Sequential(*[ResBlock(channel_num) for _ in range(nr_blocks)])


# Upsample Block
class UpBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=2, add_blur=False):
        super(UpBlock, self).__init__()

        self.shuffle = nn.ConvTranspose2d(
            in_f, out_f, kernel_size=3, stride=stride, padding=0
        )

        self.has_blur = add_blur
        if self.has_blur:
            self.blur = nn.AvgPool2d(2, 1)

    def forward(self, x):
        x = self.shuffle(x)
        if self.has_blur:
            x = self.blur(x)
        return x

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Downsampling encoder
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1,
                      padding=7 // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, 128, stride=2),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 256, stride=2),
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res = nn.Sequential(add_resblocks(256, 8))

        # Upsampling decoder
        self.up = nn.Sequential(
            UpBlock(256, 128, stride=2, add_blur=True),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            UpBlock(128, 64, stride=2, add_blur=True),
            conv3x3(64, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=7 // 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return x