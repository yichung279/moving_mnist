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
    def __init__(self, model, criterion, optimizer, lr, metric_functions=None, log_dir=None, patient=30, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.metric_functions = metric_functions
        self.performance = self.default_performance()
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None
        self.writer = SummaryWriter(log_dir)
        self.default_patient = patient
        self.patient = patient
        self.best_model = None
        self.best_valid_loss = float('inf')


    def default_performance(self):
        performance_keys = ['loss'] + [fn.__name__ for fn in self.metric_functions]
        return { key: 0 for key in performance_keys }


    def train(self, train_dataloader, valid_dataloader, n_epoch=30):
        for i in range(n_epoch):
            print(f'Epoch {i+1}:')
            self.train_epoch(train_dataloader)
            for key in self.performance:
                self.writer.add_scalar(f'{key}/train', self.performance[key], i)

            self.validate_epoch(valid_dataloader)
            for key in self.performance:
                self.writer.add_scalar(f'{key}/train', self.performance[key], i)

            if self.performance['loss'] < self.best_valid_loss:
                self.best_model = self.model
                self.best_valid_loss = self.performance['loss']
                self.patient =  self.default_patient
            else:
                self.patient -= 1
            print(self.patient)

            if self.patient == 0:
                break


    def get_model(self):
        return self.model


    def get_best_model(self):
        return self.best_model


    def train_epoch(self, train_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(train_dataloader)
        progress_bar.set_description('Train')
        for i, (x_train, y_train) in enumerate(progress_bar):
            loss = self.iterate_batch(x_train, y_train)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.performance['loss'] = round(self.running_loss/(i+1), 3)
            if self.metric_functions:
                for fn, metric in zip(self.metric_functions, self.running_metrics):
                    self.performance[fn.__name__] = round(metric/(i+1), 3)

            progress_bar.set_postfix_str(str(self.performance))


    def validate_epoch(self, valid_dataloader):
        self.running_loss = 0
        self.running_metrics = [0 for i in self.metric_functions] if self.metric_functions else None

        progress_bar = tqdm(valid_dataloader)
        progress_bar.set_description('Valid')
        for i, (x_valid, y_valid) in enumerate(progress_bar):
            with torch.no_grad():
                self.iterate_batch(x_valid, y_valid)

            self.performance['loss'] = round(self.running_loss/(i+1), 3)
            if self.metric_functions:
                for fn, metric in zip(self.metric_functions, self.running_metrics):
                    self.performance[fn.__name__] = round(metric/(i+1), 3)

            progress_bar.set_postfix_str(str(self.performance))

    def iterate_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)
        self.running_loss += loss.item()

        if self.metric_functions:
            for i, fn in enumerate(self.metric_functions):
                self.running_metrics[i] += fn(y_pred, y).item()

        return loss
