import os.path
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TinySRDataset(Dataset):
    def __init__(self, meta_info, root, num_chans=3):
        self.num_chans = num_chans
        self.filepath = pd.read_csv(meta_info, sep=",", header=None)
        self.root = root
        self.transfroms = transforms.Compose([
            
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        print(self.filepath.shape)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.root, self.filepath[0][idx].strip()))
        lr_img = Image.open(os.path.join(self.root, self.filepath[1][idx].strip()))
        # if self.num_chans==1:
        #     hr_img = hr_img.convert("L")
        #     lr_img = lr_img.convert("L")
        #lr_img = lr_img.resize(hr_img.size)
        hr_img = self.transfroms(hr_img)
        lr_img = self.transfroms(lr_img)
        return {
            "lr": lr_img,
            "hr": hr_img
        }

    def __len__(self):
        return len(self.filepath)
