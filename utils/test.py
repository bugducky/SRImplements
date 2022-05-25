from torchvision.utils import save_image
from .pair_dataset import *
import glob
from torch.utils.data import DataLoader
from torch import nn
import torch
from .pair_dataset import denormalize


def valid_sr_model(net, valid_img_dir, valid_meta_info, saved_path, mean, std):
    img_lr = []
    img_sr = []
    img_hr = []
    net.eval()
    dataset = PairDataset(valid_meta_info, valid_img_dir, mean, std, False)
    loader = DataLoader(dataset)
    counter = 0
    with torch.no_grad():
        for data in loader:
            
            img_hr = data['hr'].cuda()
            img_lr = data['lr'].cuda()
            gen_hr = net(img_lr)
            img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((img_lr, gen_hr, img_hr), -1), mean, std)
            save_image(
                img_grid, f"{saved_path}/{counter}.png", nrow=1, normalize=False
            )
            counter+=1


