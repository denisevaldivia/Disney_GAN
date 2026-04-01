import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_blocks import InvertedResBlock, ConvBlock  # keep your conv blocks

class GeneratorV2(nn.Module):
    def __init__(self, weights_path: str = None):
        """
        Generator for AnimeGANv2. If weights_path is provided, loads the weights automatically.
        Only for inference.
        """
        super(GeneratorV2, self).__init__()

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
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.conv_block4 = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 32, kernel_size=7, padding=3, stride=1, norm_type="layer"),
        )

        self.decode_blocks = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        # Load weights if provided
       
        if weights_path is not None:
            self.load_weights(weights_path)

        # Set to eval mode
        self.eval()

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

    @torch.no_grad()
    def load_weights(self, path: str, map_location=None):
        """
        Load weights from a .pth/.pt file.
        """
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        state_dict = torch.load(path, map_location=map_location)

        # remove 'module.' if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=True)
        self.eval()