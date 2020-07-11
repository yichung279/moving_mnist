#!/usr/bin/env python3
# standard imports
import math

# thrid-party imports
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


# local imports


class GeographicEncoding(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
         # shape (N, C, H, W)
        self.channel = channel
        self.height = height
        self.width = width
        self.mean = (height-1) / 2
        self.variance = self.mean * self.mean
        self.loc = torch.tensor([self.mean, self.mean])
        self.covariance_matrix = torch.tensor([[self.variance, 0], [0, self.variance]])
        self.distribution = MultivariateNormal(self.loc, self.covariance_matrix)
        self.distribution_map = self.build_distribution_map()


    def build_distribution_map(self):
        #! TODO: build map torchly
        distribution_map = torch.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                distribution_map[i][j] = self.distribution.log_prob(torch.tensor((i, j))).exp()

        return distribution_map.repeat(self.channel, 1, 1)


    def forward(self, x):
        return x + self.distribution_map


# def positional_encode(pos, shape, i=1, d=2):
#     pe =  math.sin(pos / math.pow(10000, 2*i/d))
#
#     tensor = torch.ones(())
#     return tensor.new_full(shape, pe)
#
#
if '__main__' == __name__:
    x = torch.ones((2, 64, 64))
    encoding = GeographicEncoding(2, 64, 64)
    x = encoding(x)
    # print(encoding.loc, encoding.covariance_matrix, encoding.distribution_map, sep='\n')
    print(x)
