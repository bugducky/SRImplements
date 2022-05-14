from torchvision.utils import save_image

from dataset import *
from model import *
from torch.utils.data import DataLoader
from torch.optim import SGD
import wandb
from torch import nn

LR = 0.001
MOMENTUM = 0.9
BS = 32
EPOCH = 1000
PROJECT = "srcnn"

os.makedirs("exp/training", exist_ok=True)
os.makedirs("exp/saved_models", exist_ok=True)

dataset = TinySRDataset("D:\\code\\deeplearning\\SRTinyDataset\\dataset\\bsds200\\meta_info.txt",
                        "D:\\code\\deeplearning\\SRTinyDataset\\dataset\\bsds200")
train_iter = DataLoader(dataset, batch_size=BS, shuffle=True)

net = SRCnnModel(padding=1)
optim = SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
lossfn = nn.MSELoss()

wandb.init(project=PROJECT)

Tensor = torch.cuda.FloatTensor
net = net.cuda()
batches_done = 0
for epoch in range(EPOCH + 1):
    for data in train_iter:
        img_lr = data["lr"].cuda()
        img_hr = data["hr"].cuda()
        y = net(img_lr)

        optim.zero_grad()
        loss = lossfn(img_hr, y)
        loss.backward()
        if batches_done%100==0:
            wandb.log({"mse loss": loss.item()})
            print(f"[iters: {batches_done}, loss: {loss.item()}")
        optim.step()
        batches_done += 1
        if batches_done % 1000 == 0:
            img_grid = torch.cat((img_lr, y), -1)
            img = np.array(y.cpu().detach())[0]
            cv2.imwrite("latest.png", np.transpose(img, (1, 2, 0))*255.0)
            save_image(img_grid, f"exp/training/{epoch}.png", nrow=1, normalize=False)
            image = wandb.Image(f"exp/training/{epoch}.png", caption=f"{batches_done}.png")
            wandb.log({"images": image})
