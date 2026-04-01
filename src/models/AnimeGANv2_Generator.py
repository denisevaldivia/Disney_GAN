import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# NOTE!!! This code is an adaptation form the original repo, we are not owners nor the creators behind the design, we justa dapted it to our needs
# Original: https://github.com/ptran1203/pytorch-animeGAN/tree/master

def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.02)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm_type, channels):
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels)
    elif norm_type == "layer":
        # return LayerNorm2d
        return nn.GroupNorm(num_groups=1, num_channels=channels, affine=True)
        # return partial(nn.GroupNorm, 1, out_ch, 1e-5, True)
    else:
        raise ValueError(norm_type)

class ConvBlock(nn.Module):
    """Stack of Conv2D + Norm + LeakyReLU"""
    def __init__(
        self,
        channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        padding=1,
        bias=False,
        norm_type="instance"
    ):
        super(ConvBlock, self).__init__()

        # if kernel_size == 3 and stride == 1:
        #     self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        # elif kernel_size == 7 and stride == 1:
        #     self.pad = nn.ReflectionPad2d((3, 3, 3, 3))
        # elif stride == 2:
        #     self.pad = nn.ReflectionPad2d((0, 1, 1, 0))
        # else:
        #     self.pad = None
        
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=0,
            bias=bias
        )
        self.ins_norm = get_norm(norm_type, out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        # initialize_weights(self)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)
        return out


class InvertedResBlock(nn.Module):
    def __init__(
        self,
        channels=256,
        out_channels=256,
        expand_ratio=2,
        norm_type="instance",
    ):
        super(InvertedResBlock, self).__init__()
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(
            channels,
            bottleneck_dim,
            kernel_size=1,
            padding=0,
            norm_type=norm_type,
            bias=False
        )
        self.conv_block2 = ConvBlock(
            bottleneck_dim,
            bottleneck_dim,
            groups=bottleneck_dim,
            norm_type=norm_type,
            bias=True
        )
        self.conv = nn.Conv2d(
            bottleneck_dim,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.norm = get_norm(norm_type, out_channels)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.conv_block2(out)
        # out = self.activation(out)
        out = self.conv(out)
        out = self.norm(out)

        if out.shape[1] != x.shape[1]:
            # Only concate if same shape
            return out
        return out + x

class GeneratorV2(nn.Module):
    def __init__(self, dataset=''):
        super(GeneratorV2, self).__init__()
        self.name = f'{self.__class__.__name__}_{dataset}'

        self.conv_block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=1, padding=3, norm_type="layer"),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=(0, 1, 0, 1), norm_type="layer"),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.conv_block2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=(0, 1, 0, 1), norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.res_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
            InvertedResBlock(128, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            ConvBlock(256, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.conv_block3 = nn.Sequential(
            # UpConvLNormLReLU(128, 128, norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.conv_block4 = nn.Sequential(
            # UpConvLNormLReLU(128, 64, norm_type="layer"),
            ConvBlock(128, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 32, kernel_size=7, padding=3, stride=1, norm_type="layer"),
        )

        self.decode_blocks = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.res_blocks(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv_block3(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv_block4(out)
        img = self.decode_blocks(out)

        return img