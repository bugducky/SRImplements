import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
import os
import random


# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors, mean, std, max_value=255):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, max_value)


class PairDataset(Dataset):
    def __init__(self, root, meta_info, in_chans=3):
        self.filepath = pd.read_csv(meta_info, sep=",", header=None)
        self.root = root
        self.in_chans = in_chans
        self.color_flag = {1: cv2.IMREAD_GRAYSCALE, 3: cv2.IMREAD_COLOR}
        self.transfroms = A.Compose(
            [A.RandomRotate90(), A.VerticalFlip(), A.HorizontalFlip()], additional_targets={"image0": "image"},
        )

    def __getitem__(self, idx):
        lr_img = cv2.imread(os.path.join(self.root, self.filepath[1][idx].strip()), self.color_flag[self.in_chans])
        hr_img = cv2.imread(os.path.join(self.root, self.filepath[0][idx].strip()), self.color_flag[self.in_chans])
        aug = self.transfroms(image=hr_img, image0=lr_img)
        hr_img = torch.FloatTensor(aug["image"] / 255.0)
        lr_img = torch.FloatTensor(aug["image0"] / 255.0)
        if self.in_chans == 1:
            hr_img, lr_img = torch.unsqueeze(hr_img, 0), torch.unsqueeze(lr_img, 0)
        return {"lr": lr_img, "hr": hr_img}

    def __len__(self):
        return len(self.filepath)
