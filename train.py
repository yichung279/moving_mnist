#!/usr/bin/env python3


# standard imports

# thrid-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import mean_squared_error


# local imports
from dataset import MMnistDataset
from trainer import Trainer


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(10, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv7_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 10, 3, padding=1)


    def forward(self, x):
        x = torch.reshape(x, (-1, 10, 64,64))

        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x1 = F.relu(x)
        x = self.maxpool1(x1)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x2 = F.relu(x)
        x = self.maxpool2(x2)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x3 = F.relu(x)
        x = self.maxpool3(x3)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x4 = F.relu(x)
        x = self.maxpool4(x4)

        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x5 = self.up5(x)

        x = torch.cat([x5, x4], dim=1)
        x = self.conv6_1(x)
        x = F.relu(x)
        x = self.conv6_2(x)
        x = F.relu(x)
        x6 = self.up6(x)

        x = torch.cat([x6, x3], dim=1)
        x = self.conv7_1(x)
        x = F.relu(x)
        x = self.conv7_2(x)
        x = F.relu(x)
        x7 = self.up7(x)

        x = torch.cat([x7, x2], dim=1)
        x = self.conv8_1(x)
        x = F.relu(x)
        x = self.conv8_2(x)
        x = F.relu(x)
        x8 = self.up8(x)

        x = torch.cat([x8, x1], dim=1)
        x = self.conv9_1(x)
        x = F.relu(x)
        x = self.conv9_2(x)
        x = torch.sigmoid(x)

        x = torch.reshape(x, (-1, 10, 1, 64,64))

        return x

def mse1(y_pred, y):
    e = y_pred.sub(y)
    se = torch.pow(e, 2)
    mse = torch.mean(se)
    return mse

if '__main__' == __name__:

    train_dataset = MMnistDataset('/local/mnist_data/train/', 100)
    train_dataloader = DataLoader(train_dataset,
            batch_size=train_dataset.batch_size,
            collate_fn=None)

    valid_dataset = MMnistDataset('/local/mnist_data/valid/', 100)
    valid_dataloader = DataLoader(valid_dataset,
            batch_size=valid_dataset.batch_size,
            collate_fn=None)

    test_dataset = MMnistDataset('/local/mnist_data/test/', 100)
    test_dataloader = DataLoader(test_dataset,
            batch_size=test_dataset.batch_size,
            collate_fn=None)

    model = UNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam
    summary(model, (10, 1, 64, 64), device='cpu')

    trainer =  Trainer(model, criterion, optimizer, lr=1e-4, with_gpu=True, metric_functions=[mse1])
    trainer.train(train_dataloader, valid_dataloader, 1)

    model = trainer.get_model()

    y_pred = np.array([])
    y_true = np.array([])
    for x_test, y_test in test_dataloader:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            output = model(x_test)

            output = output.reshape((-1, ))
            y_test = y_test.reshape((-1, ))

            output = output.cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()

            y_pred = np.append(y_pred, output)
            y_true = np.append(y_true, y_test)
    print(mean_squared_error(y_true, y_pred))


