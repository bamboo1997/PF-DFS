import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveCNN(nn.Module):
    def __init__(self, cfg, init_weights: bool = True):
        super().__init__()
        if "mnist" in cfg.dataset.name:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3)

        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(512)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)

        self.pool = nn.MaxPool2d((2, 2))
        if "mnist" == cfg.dataset.name:
            self.fc1 = nn.Linear(512 * 1 * 1, 512)
        else:
            _img_s = math.floor((cfg.dataset.image_size - 2) / 2)
            _img_s = math.floor((_img_s - 2) / 2)
            _img_s = math.floor((_img_s - 2) / 2)
            self.fc1 = nn.Linear(512 * _img_s * _img_s, 512)

        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, cfg.dataset.numof_classes)

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
