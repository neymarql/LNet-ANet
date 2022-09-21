# 作者：钱隆
# 时间：2022/9/1 23:10


import torch.nn as nn
from sklearn.svm import LinearSVC
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import os
from PIL import Image
from torch.utils import data
import time
import csv


class LNeto(nn.module):
    def __init__(self, num_class=1000):
        super(LNeto, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.downsample(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        out += identity
        return out


class LNets(nn.module):
    def __init__(self):
        super(LNets, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.downsample(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        out += identity
        return out


class ANet(nn.module):
    def __init__(self, num_class=1000):
        super(ANet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1, padding=0),
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


class Dataset_Csv(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, folders, labels, transform=None):
        """Initialization"""
        # self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.folders)

    def read_images(self, path, use_transform):
        image = Image.open(path)
        if use_transform is not None:
            image = use_transform(image)
        return image

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


def make_weights_for_balanced_classes(train_dataset, stage='train'):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight
