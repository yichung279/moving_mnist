#!/usr/bin/env python3


# standard imports

# thrid-party imports
from tqdm import tqdm
import torch

# local imports


class Trainer():
    def __init__(self, model, criterion, optimizer, lr=1e-4, metric_functions=None, callback=None, use_gpu=True):
        self.device = torch.device('cpu')
        if use_gpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.metric_functions = metric_functions
        self.callback = callback
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None


    def train(self, train_dataloader, valid_dataloader, n_epoch=30):
        for i in range(n_epoch):
            print(f'Epoch {i+1}:')
            train_loss, train_metrics = self.train_epoch(train_dataloader)
            valid_loss, valid_metrics = self.validate_epoch(valid_dataloader)


    def train_epoch(self, train_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(train_dataloader)
        for i, (x_train, y_train) in enumerate(progress_bar):
            loss = self.evalute_batch(x_train, y_train)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            mean_loss = self.running_loss / (i+1)
            mean_metrics = [metric / (i+1) for metric in self.running_metrics] if self.metric_functions else None

            train_metrics = [f'{metric:.2f}' for metric in mean_metrics] if self.metric_functions else None
            progress_bar.set_postfix_str(f'train_loss: {mean_loss:.2f}, train_metrics: {train_metrics}')

        return mean_loss, mean_metrics


    def validate_epoch(self, valid_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(valid_dataloader)
        for i, (x_valid, y_valid) in enumerate(progress_bar):
            with torch.no_grad():
                self.evalute_batch(x_valid, y_valid)

            mean_loss = self.running_loss / (i+1)
            mean_metrics = [metric / (i+1) for metric in self.running_metrics] if self.metric_functions else None

            valid_metrics = [f'{metric:.2f}' for metric in mean_metrics] if self.metric_functions else None
            progress_bar.set_postfix_str(f'valid_loss: {mean_loss:.2f}, valid_metrics: {valid_metrics}')

        return mean_loss, mean_metrics

    def evalute_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)
        self.running_loss += loss

        if self.metric_functions:
            for i, fn in enumerate(self.metric_functions):
                self.running_metrics[i] += fn(y_pred, y)

        return loss
