from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import wandb
from torch import nn
import torch
import numpy as np
import os
import sys
from rcan import RCANModel

from implements.utils import TinySRDataset

LR = 1e-5
BS = 8
EPOCH = 100
PROJECT = "rcan"
USE_WANDB = False

os.makedirs(f"experiments/{PROJECT}/training", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/saved_models", exist_ok=True)

dataset = TinySRDataset(
    "D:\\code\\deeplearning\\dataset\\bsds200\\meta_info.txt", "D:\\code\\deeplearning\\dataset\\bsds200"
)
train_iter = DataLoader(dataset, batch_size=BS, shuffle=True)

net = RCANModel(3, 3)
optim = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
lossfn = nn.L1Loss()

if USE_WANDB:
    wandb.init(project=PROJECT)

net = net.cuda()


def main():
    batches_done = 0
    for epoch in range(EPOCH + 1):
        for data in train_iter:
            img_lr = data["lr"].cuda()
            img_hr = data["hr"].cuda()
            y = net(img_lr)

            optim.zero_grad()
            loss = lossfn(img_hr, y)
            loss.backward()
            if batches_done % 20 == 0:
                if USE_WANDB:
                    wandb.log({"mse loss": loss.item()})
                print(f"[iters: {batches_done}, loss: {loss.item()}")
            optim.step()
            batches_done += 1
            if batches_done % 500 == 0:
                img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
                y = torch.clamp(y, 0, 255)
                # img_lr = torch.clamp(img_lr, 0, 255)
                img_grid = torch.cat((img_lr, y, img_hr), -1)
                img = np.array(y.cpu().detach())[0]
                save_image(img_grid, f"experiments/{PROJECT}/training/{batches_done}.png", nrow=2, normalize=False)
                if USE_WANDB:
                    image = wandb.Image(f"experiments/{PROJECT}/training/{batches_done}.png",
                                        caption=f"{batches_done}.png")
                    wandb.log({"images": image})
            if batches_done % 5000 == 0:
                torch.save(net.state_dict(), f"experiments/{PROJECT}/saved_models/iter_{batches_done}.pth")


if __name__ == '__main__':
    main()
