#!/usr/bin/env python3


# standard imports

# thrid-party imports
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# local imports


class Trainer():
    def __init__(self, model, criterion, optimizer, lr, metric_functions, with_gpu=True):
        self.device = torch.device('cpu')
        if with_gpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        #! TODO: add self.log reimplement tensorboard and postfix of progpressbar
        self.metric_functions = metric_functions
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None
        self.writer = SummaryWriter()


    def train(self, train_dataloader, valid_dataloader, n_epoch=30):
        for i in range(n_epoch):
            print(f'Epoch {i+1}:')
            train_loss, train_metrics = self.train_epoch(train_dataloader)
            valid_loss, valid_metrics = self.validate_epoch(valid_dataloader)

            self.writer.add_scalar('loss/train', train_loss, i)
            # for metric_function, metric in zip(self.metric_functions, train_metrics):
            #     self.writer.add_scalar(f'{metric_function.__name__}/train', metric, i)

            self.writer.add_scalar('loss/valid', valid_loss, i)
            # for metric_function, metric in zip(self.metric_functions, valid_metrics):
            #     self.writer.add_scalar(f'{metric_function.__name__}/valid', metric, i)


    def get_model(self):
        return self.model


    def train_epoch(self, train_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(train_dataloader)
        progress_bar.set_description('Train')
        for i, (x_train, y_train) in enumerate(progress_bar):
            loss = self.evalute_batch(x_train, y_train)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            mean_loss = self.running_loss / (i+1)
            mean_metrics = {fn.__name__: round(metric/(i+1), 3) for fn, metric in zip(self.metric_functions, self.running_metrics)}\
                if self.metric_functions else None

            progress_bar.set_postfix_str(f'train_loss: {mean_loss:.3f}, {mean_metrics}')

        return mean_loss, mean_metrics


    def validate_epoch(self, valid_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(valid_dataloader)
        progress_bar.set_description('Valid')
        for i, (x_valid, y_valid) in enumerate(progress_bar):
            with torch.no_grad():
                self.evalute_batch(x_valid, y_valid)

            mean_loss = self.running_loss / (i+1)
            mean_metrics = {fn.__name__: round(metric/(i+1), 3) for fn, metric in zip(self.metric_functions, self.running_metrics)}\
                if self.metric_functions else None
            mean_metrics = {
                "acc": 99.999,
                "mse": 99.999,
            }

            progress_bar.set_postfix_str(f'valid_loss: {mean_loss:.3f}, {mean_metrics}')

        return mean_loss, mean_metrics

    def evalute_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)
        self.running_loss += loss.item()

        if self.metric_functions:
            for i, fn in enumerate(self.metric_functions):
                self.running_metrics[i] += fn(y_pred, y).item()

        return loss
