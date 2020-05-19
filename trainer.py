#!/usr/bin/env python3


# standard imports

# thrid-party imports
from tqdm import tqdm
import torch

# local imports


class Trainer():
    def __init__(self, model, criterion, optimizer, lr, use_gpu=True):
        self.device = torch.device("cpu")
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)


    def train(self, train_dataloader, valid_dataloader, n_epoch=30):
        for i in range(n_epoch):
            self.train_epoch(train_dataloader, valid_dataloader)


    def train_epoch(self, train_dataloader, valid_dataloader):
        progress_bar = tqdm(train_dataloader)
        for i, (train_x, train_y) in enumerate(progress_bar):

            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)

            y_pred = self.model(train_x)

            loss = self.criterion(y_pred, train_y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


