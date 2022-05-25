from torchvision.utils import save_image


from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import wandb
from torch import nn
import numpy as np
import torch

from utils import *
from models import *

LR = 2e-4
BS = 4

PROJECT = "sun_train_swinir_adv_20220522"
use_wandb = False


in_chans = 1
n_cpu = 0
lr = 2e-4
num_warm_up = -1
batchsize = 4
iter_num = 20e4
log_interval = 10
sample_interval = 5e3
checkpoint_interval = 1e4
lr_size = 64
upscale = 4
val_interval = 1e4

lambda_pixel = 1.0
lambda_adv = 0.1


swin_model_path = "experiments\swinir\saved_models\iter_20000.pth"

os.makedirs(f"experiments/{PROJECT}/training", exist_ok=True)
os.makedirs(f"experiments/{PROJECT}/saved_models", exist_ok=True)

dataset_root = "D:\\code\\deeplearning\\dataset"
data_dir = os.path.join(dataset_root, "sun/patch_256")
meta_txt = os.path.join(dataset_root, "sun/patch_256/meta_info.txt")

dataset = PairDataset(data_dir, meta_txt, 1)

train_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=n_cpu)

net_g = SwinIRModel(
    upscale=upscale,
    img_size=(lr_size, lr_size),
    in_chans=in_chans,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6],
    embed_dim=60,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)

net_g.load_state_dict(torch.load(swin_model_path))

net_d = UNetDiscriminatorSN(in_chans)
net_fe = Vgg19FeatureExtractor(in_chans)
net_fe.eval()

optimizer_G = Adam(net_g.parameters(), lr=LR, betas=[0.9, 0.99])
optimizer_D = Adam(net_d.parameters(), lr=LR, betas=[0.9, 0.99])

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

if use_wandb:
    wandb.init(project=PROJECT)

net_g = net_g.cuda()
net_d = net_d.cuda()
net_fe = net_fe.cuda()

batches_done = 0
while True:
    for data in train_iter:
        batches_done += 1
        img_lr, img_hr = data["lr"].cuda(), data["hr"].cuda()

        net_d.eval()
        for p in net_d.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()
        gen_hr = net_g(img_lr)
        loss_pixel = criterion_pixel(gen_hr, img_hr)

        if batches_done < num_warm_up:
            loss_pixel.backward()
            optimizer_G.step()
            if batches_done % log_interval == 0:
                if use_wandb:
                    wandb.log({"G_loss_pixel": loss_pixel.item()})
                print("[iters: {}] loss_pixel: {:.4f}]".format(batches_done, loss_pixel.item()))
            continue

        pred_real = net_d(img_hr).detach()
        pred_fake = net_d(gen_hr)

        valid = torch.tensor(np.ones(img_hr.shape), requires_grad=False).cuda()
        fake = torch.tensor(np.zeros(img_hr.shape), requires_grad=False).cuda()

        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        gen_features = net_fe(gen_hr)
        real_features = net_fe(img_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        net_d.train()
        for p in net_d.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        pred_real = net_d(img_hr)
        pred_fake = net_d(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        if batches_done % log_interval == 0:
            if use_wandb:
                wandb.log({
                        "loss_d": loss_D.item(),
                        "loss_g": loss_G.item(),
                        "loss_pixel": loss_pixel.item(),
                        "loss_content": loss_content.item(),
                        "loss_adv": loss_GAN.item(),
                })

            print("[iters: {:<6}] loss_g: {:.4f}, loss_d: {:.4f}, loss_pixel: {:.4f}], loss_content: {:.4f}, loss_adv: {:.4f}".format(
                    batches_done,
                    loss_G.item(),
                    loss_D.item(),
                    loss_pixel.item(),
                    loss_content.item(),
                    loss_GAN.item(),
            ))
        
        # if batches_done % val_interval == 0:
        #     pass

        if batches_done % sample_interval == 0:
            img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
            img_grid = torch.cat((img_lr, gen_hr, img_hr), -1)
            # img_grid = denormalize(torch.cat((img_lr, gen_hr, img_hr), -1), mean, std)
            save_image(
                img_grid, f"experiments/{PROJECT}/training/{batches_done}.png", nrow=2, normalize=False
            )
            if use_wandb:
                image = wandb.Image(
                    f"experiments/{PROJECT}/training/{batches_done}.png", caption=f"{batches_done}.png"
                )
                wandb.log({"images": image})

        if batches_done % checkpoint_interval == 0:
            torch.save(net_g.state_dict(), f"experiments/{PROJECT}/saved_models/iter_{batches_done}.pth")

        if iter_num == batches_done:
            torch.save(net_g.state_dict(), f"experiments/{PROJECT}/saved_models/iter_latest.pth")
            print("finished!")
            exit(0)
