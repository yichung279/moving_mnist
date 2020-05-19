#!/usr/bin/env python3


# standard imports

# thrid-party imports
from math import ceil
import os

# local imports
import numpy as np
from torch.utils.data import IterableDataset


class MMnistIterator:
    def __init__(self, dirname):
        self.dirname = dirname
        self.fnames = iter(sorted(os.listdir(self.dirname)))
        self.chunk = []
        self.chunk_iterator = iter(self.chunk)


    def __iter__(self):
        return self


    def __next__(self):
        if self.chunk_iterator.__length_hint__() == 0:
            path = os.path.join(self.dirname, next(self.fnames))
            chunk = np.load(path)
            self.chunk = np.swapaxes(chunk, 0, 1)
            self.chunk_iterator = iter(self.chunk)
        data = next(self.chunk_iterator)
        return data[:10], data[10:]


class MMnistDataset(IterableDataset):
    def __init__(self, dirname, batch_size=1):
        super(MMnistDataset, self).__init__()
        self.dirname = dirname
        self.batch_size = batch_size


    def __iter__(self):
        return MMnistIterator(self.dirname)


    def __len__(self):
        return ceil(len(os.listdir(self.dirname)) * 1000 / self.batch_size)
