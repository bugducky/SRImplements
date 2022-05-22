import os.path
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch


# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors, mean, std, max_value=255):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, max_value)


class TinySRDataset(Dataset):
    def __init__(self, meta_info, root, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_train=True):
        self.filepath = pd.read_csv(meta_info, sep=",", header=None)
        self.root = root
        if is_train:
            self.transfroms = A.Compose(
                [
                    A.RandomRotate90(),
                    A.Flip(),
                    A.Normalize(mean, std),
                    ToTensorV2()
                ],
                additional_targets={"image0": "image"}
            )
        else:
            self.transfroms = A.Compose(
                [
                    A.Normalize(mean, std),
                    ToTensorV2()
                ],
                additional_targets={"image0": "image"}
            )
        print(self.filepath.shape)

    def __getitem__(self, idx):
        lr_img = cv2.imread(os.path.join(self.root, self.filepath[1][idx].strip()))
        hr_img = cv2.imread(os.path.join(self.root, self.filepath[0][idx].strip()))
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        aug = self.transfroms(image=hr_img, image0=lr_img)
        hr_img = torch.FloatTensor(aug["image"])
        lr_img = torch.FloatTensor(aug["image0"])

        return {"lr": lr_img, "hr": hr_img}

    def __len__(self):
        return len(self.filepath)
