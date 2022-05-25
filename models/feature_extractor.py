import torch
from torch import nn
from torchvision.models import vgg19, resnet50


class Vgg19FeatureExtractor(nn.Module):
    def __init__(self, in_chans=3) -> None:
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        module_list = list(vgg19_model.features.children())[:35]
        if in_chans == 1:
            module_list[0] = nn.Conv2d(1, 64, 3, 1, 1)
        self.feature_extractor = nn.Sequential(*module_list)

    def forward(self, img):
        return self.feature_extractor(img)