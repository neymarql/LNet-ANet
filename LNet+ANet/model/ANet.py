# 作者：钱隆
# 时间：2022/8/24 20:38

import torch.nn as nn
from sklearn.svm import LinearSVC
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import os


class ANet(nn.module):
    def __init__(self,num_class=1000):
        super(ANet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=2, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=80, kernel_size=3, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(80, num_class)
        )

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = out.view(80, -1)
        out = self.layer5(out)
        return out
