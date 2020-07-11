#!/usr/bin/env python3


# standard imports


# thrid-party imports
import torch
import torch.nn.functional as F
from torch import nn


# local imports
from transformer.Encoding import GeographicEncoding
from transformer.Sublayers import MultiHeadImageAttentionBlock


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x):
        x = self.encoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self. = MultiHeadImageAttentionBlock()
        self. = FeedForwardBlock()
