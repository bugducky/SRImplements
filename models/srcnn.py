import torch
from torch import nn
import cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F


class SRCnnModel(nn.Module):
    """
    Implementation details. Training is performed on the 91-
    image dataset, and testing is conducted on the Set5 [2]. The
    network settings are: c1 = 3, f1 = 9, f2 = 1, f3 = 5, n1 = 64,
    and n2 = 32.
    """

    def __init__(self, num_channels=3, padding=0):
        super(SRCnnModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 64, 9, padding=4 * padding, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, num_channels, 5, padding=2 * padding, padding_mode='replicate')
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

    def init_weight(self):
        self.conv1.weight.data.normal_(mean=0.0, std=0.001)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(mean=0.0, std=0.001)
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.normal_(mean=0.0, std=0.001)
        self.conv3.bias.data.zero_()

