from torchvision.utils import save_image

from utils import *
from models import SRResNet, UNetDiscriminatorSN, FeatureExtractor
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import wandb
from torch import nn
import numpy as np
import torch

LR = 2e-4
BS = 4

ITERS_NUM = 11000
EPOCH = ITERS_NUM
PROJECT = "srgan"
USE_WANDB = False
WARM_G = 1000
print_interval = 100
save_interval = 5000
sample_interval = 500
val_interval = 2000
lambda_pixel = 1.0
lambda_adv = 0.01

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

os.makedirs(f"experiments/{PROJECT}/training", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/saved_models", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/valid_img", exist_ok=True)

dataset = TinySRDataset(
    "D:\\code\\deeplearning\\dataset\\bsds200\\meta_info.txt",
    "D:\\code\\deeplearning\\dataset\\bsds200",
    mean,
    std,
)

train_iter = DataLoader(dataset, batch_size=BS, shuffle=True)

net_g = SRResNet()

net_g.load_state_dict(torch.load("experiments/srresnet/saved_models/iter_10000.pth"))

net_d = UNetDiscriminatorSN(3)
net_fe = FeatureExtractor()
net_fe.eval()

optimizer_G = Adam(net_g.parameters(), lr=LR, betas=[0.9, 0.99])
optimizer_D = Adam(net_d.parameters(), lr=LR, betas=[0.9, 0.99])

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

if USE_WANDB:
    wandb.init(project=PROJECT)

net_g = net_g.cuda()
net_d = net_d.cuda()
net_fe = net_fe.cuda()

batches_done = 0
while True:
    for data in train_iter:

        batches_done += 1

        img_lr = data["lr"].cuda()
        img_hr = data["hr"].cuda()
        valid = torch.tensor(np.ones(img_hr.shape), requires_grad=False).cuda()
        fake = torch.tensor(np.zeros(img_hr.shape), requires_grad=False).cuda()

        optimizer_G.zero_grad()
        gen_hr = net_g(img_lr)
        loss_pixel = criterion_pixel(gen_hr, img_hr)

        if batches_done < WARM_G:
            loss_pixel.backward()
            optimizer_G.step()
            if batches_done % print_interval == 0:
                if USE_WANDB:
                    wandb.log({"G_loss_pixel": loss_pixel.item()})
                print("[iters: {}] loss_pixel: {:.4f}]".format(batches_done, loss_pixel.item()))
            continue

        pred_real = net_d(img_hr).detach()
        pred_fake = net_d(gen_hr)

        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        gen_features = net_fe(gen_hr)
        real_features = net_fe(img_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        pred_real = net_d(img_hr)
        pred_fake = net_d(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        if batches_done % print_interval == 0:
            if USE_WANDB:
                wandb.log(
                    {
                        "loss_d": loss_D.item(),
                        "loss_g": loss_G.item(),
                        "loss_pixel": loss_pixel.item(),
                        "loss_content": loss_content.item(),
                        "loss_adv": loss_GAN.item(),
                    }
                )

            print(
                "[iters: {:<6}] loss_g: {:.4f}, loss_d: {:.4f}, \n\tloss_pixel: {:.4f}], loss_content: {:.4f}, loss_adv: {:.4f}".format(
                    batches_done,
                    loss_G.item(),
                    loss_D.item(),
                    loss_pixel.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                )
            )
        if batches_done % val_interval == 0:
            valid_sr_model(
                net_g,
                "D:\\code\\deeplearning\\dataset\\set14_val",
                "D:\\code\\deeplearning\\dataset\\set14_val\\meta_info.txt",
                f"experiments/{PROJECT}/valid_img",
                mean, std
            )

        if batches_done % sample_interval == 0:
            img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((img_lr, gen_hr, img_hr), -1), mean, std)
            save_image(
                img_grid, f"experiments/{PROJECT}/training/{batches_done}.png", nrow=2, normalize=False
            )
            if USE_WANDB:
                image = wandb.Image(
                    f"experiments/{PROJECT}/training/{batches_done}.png", caption=f"{batches_done}.png"
                )
                wandb.log({"images": image})

        if batches_done % save_interval == 0:
            torch.save(net_g.state_dict(), f"experiments/{PROJECT}/saved_models/iter_{batches_done}.pth")

    if ITERS_NUM == batches_done:
        torch.save(net_g.state_dict(), f"experiments/{PROJECT}/saved_models/iter_latest.pth")
        break
