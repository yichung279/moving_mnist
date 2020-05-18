#!/usr/bin/env python3
import os
import numpy as np

from torch.utils.data import IterableDataset


class MnistIterator:
    def __init__(self, dirname):
        self.dirname = dirname
        self.fnames = iter(sorted(os.listdir(self.dirname)))
        # for i in self.fnames:
        #     print(i)
        self.chunk = []
        self.chunk_iterator = iter(self.chunk)

    def __iter__(self):
        return self

    def __next__(self):
        if self.chunk_iterator.__length_hint__() == 0:
            # Load all tensors onto GPU 0
            path = os.path.join(self.dirname, next(self.fnames))
            # self.chunk = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
            chunk = np.load(path)
            chunk = np.swapaxes(chunk, 0, 1)
            self.chunk = chunk.reshape((-1, 20, 64, 64))
            self.chunk_iterator = iter(self.chunk)
        return next(self.chunk_iterator)


class MnistDataset(IterableDataset):
    def __init__(self, dirname):
        super(MnistDataset, self).__init__()
        self.dirname = dirname

    def __iter__(self):
        return MnistIterator(self.dirname)
