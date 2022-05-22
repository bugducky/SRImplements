from torchvision.utils import save_image

from utils import *
from models import SRResNet
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import wandb
from torch import nn
import numpy as np

LR = 2e-4
BS = 16

ITERS_NUM = 1e5
EPOCH = ITERS_NUM
PROJECT = "srresnet"
USE_WANDB = False

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

os.makedirs(f"experiments/{PROJECT}/training", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/saved_models", exist_ok=True)

dataset = TinySRDataset(
    "D:\\code\\deeplearning\\dataset\\bsds200\\meta_info.txt",
    "D:\\code\\deeplearning\\dataset\\bsds200", mean, std
)

train_iter = DataLoader(dataset, batch_size=BS, shuffle=True)

net = SRResNet()
optim = Adam(net.parameters(), lr=LR, betas=[0.9, 0.99])
lossfn = nn.MSELoss()

if USE_WANDB:
    wandb.init(project=PROJECT)

net = net.cuda()

batches_done = 0
# for epoch in range(EPOCH + 1):
while True:
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
        if batches_done % 200 == 0:
            img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((img_lr, y, img_hr), -1), mean, std)
            img = np.array(y.cpu().detach())[0]
            save_image(img_grid, f"experiments/{PROJECT}/training/{batches_done}.png", nrow=2, normalize=False)
            if USE_WANDB:
                image = wandb.Image(f"experiments/{PROJECT}/training/{batches_done}.png", caption=f"{batches_done}.png")
                wandb.log({"images": image})
        if batches_done % 5000 == 0:
            torch.save(net.state_dict(), f"experiments/{PROJECT}/saved_models/iter_{batches_done}.pth")
    if ITERS_NUM == batches_done:
        torch.save(net.state_dict(), f"experiments/{PROJECT}/saved_models/iter_latest.pth")
        break
