import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import cv2
import os
import numpy as np
from .transformations import get_default_transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    """Image dataset.""" 

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.root_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.root_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.root_files[idx]))
        if self.transform:
            img = self.transform(img)

        return img

def get_dataloader(path="./datasets/real_images", size=256, bs=64, trfs=None, flip=.005):
    "If no transforms is specified use default transforms"

    if not trfs:
        trfs = get_default_transforms(size=size)
    dset = ImageDataset(path, transform=trfs)
    return DataLoader(dset, batch_size=bs, num_workers=4, drop_last=True, shuffle=True)