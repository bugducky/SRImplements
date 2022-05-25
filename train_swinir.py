from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim import SGD, Adam
import wandb
from torch import nn
import torch
import numpy as np
import os
import sys
from torch.nn import functional as F

from models import *
from utils import *

PROJECT = "sun_train_swinir_20220522"
use_wandb = False

lr = 2e-4
batchsize = 4
# iter_num = 10e4
iter_num = 100
log_interval = 10
sample_interval = 1000
checkpoint_interval = 1e4
lr_size = 64
upscale = 4

os.makedirs(f"experiments/{PROJECT}/training", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/saved_models", exist_ok=True)

dataset_root = "D:\\code\\deeplearning\\dataset"
data_dir = os.path.join(dataset_root, "sun/patch_256")
meta_txt = os.path.join(dataset_root, "sun/patch_256/meta_info.txt")

dataset = PairDataset(data_dir, meta_txt, 1)

train_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True)

net = SwinIRModel(upscale=upscale, img_size=(lr_size, lr_size), in_chans=1,
                  window_size=8, img_range=1., depths=[6, 6, 6, 6],
                  embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                  upsampler='pixelshuffle', resi_connection="1conv")

optim = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
# lr_desc = StepLR(optim, step_size=2e4, gamma=0.5)
lr_desc = MultiStepLR(optim, [iter_num * 0.5, iter_num * 0.75, iter_num * 0.9], gamma=0.5)
lossfn = nn.L1Loss()

if use_wandb:
    wandb.init(project=PROJECT)

net = net.cuda()
batches_done = 0
while True:
    for data in train_iter:
        batches_done += 1
        img_lr = data["lr"].cuda()
        img_hr = data["hr"].cuda()
        y = net(img_lr)

        optim.zero_grad()
        loss = lossfn(img_hr, y)

        if batches_done % log_interval == 0:
            if use_wandb:
                wandb.log({"loss": loss.item()})
            print(f"[iters: {batches_done}, loss: {loss.item()}")

        loss.backward()
        optim.step()
        lr_desc.step()

        if batches_done % sample_interval == 0:
            img_lr = F.interpolate(img_lr, scale_factor=4)
            # img_grid = denormalize(torch.cat((img_lr, y, img_hr), -1), mean, std)
            img_grid = torch.cat((img_lr, y, img_hr), -1)
            img = np.array(y.cpu().detach())[0]
            save_image(
                img_grid, f"experiments/{PROJECT}/training/{batches_done}.png", nrow=2, normalize=True
            )
            if use_wandb:
                image = wandb.Image(
                    f"experiments/{PROJECT}/training/{batches_done}.png", caption=f"{batches_done}.png"
                )
                wandb.log({"image": image})
        if batches_done % checkpoint_interval == 0:
            torch.save(net.state_dict(), f"experiments/{PROJECT}/saved_models/iter_{batches_done}.pth")
        if iter_num == batches_done:
            torch.save(net.state_dict(), f"experiments/{PROJECT}/saved_models/iter_latest.pth")
            exit(0)
