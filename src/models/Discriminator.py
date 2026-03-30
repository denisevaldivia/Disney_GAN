import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# Basic conv block
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with reflection padding"""
#     return nn.Sequential(
#         nn.ReflectionPad2d(1),
#         nn.Conv2d(in_planes, out_planes, kernel_size=3,
#                   stride=stride, padding=0, bias=False)
#     )
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=3,
                      stride=stride,
                      padding=0,
                      bias=False)
    )

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(3, 32, stride=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(32, 64, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(64, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(128, 128, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(128, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(256, 1, stride=1)  # PatchGAN output
        )

    def forward(self, x):
        return self.conv(x)