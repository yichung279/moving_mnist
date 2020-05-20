#!/usr/bin/env python3


# standard imports

# thrid-party imports
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# local imports
from dataset import MMnistDataset
from trainer import Trainer


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv3_1 = nn.Conv2d(128+64, 64, 3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 10, 3, padding=1)


    def forward(self, x):
        x = torch.reshape(x, (-1, 10, 64,64))

        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x1 = F.relu(x)
        x = self.maxpool(x1)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x2 = self.up(x)

        x = torch.cat([x2, x1], dim=1)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = torch.sigmoid(x)
        x = torch.reshape(x, (-1, 10, 1, 64,64))

        return x

def mse1(y_pred, y):
    e = y_pred.sub(y)
    se = torch.pow(e, 2)
    mse = torch.mean(se)
    return mse

if '__main__' == __name__:

    train_dataset = MMnistDataset('/local/mnist_data/train/', 32)
    train_dataloader = DataLoader(train_dataset,
            batch_size=train_dataset.batch_size,
            collate_fn=None)

    valid_dataset = MMnistDataset('/local/mnist_data/valid/', 32)
    valid_dataloader = DataLoader(valid_dataset,
            batch_size=valid_dataset.batch_size,
            collate_fn=None)

    model = UNet()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD

    trainer =  Trainer(model, criterion, optimizer, lr=1e-4, use_gpu=True, metric_functions=[mse1])
    trainer.train(train_dataloader, valid_dataloader, 10)
