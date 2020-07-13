#!/usr/bin/env python3


# standard imports


# thrid-party imports
import torch
import torch.nn.functional as F
from torch import nn


# local imports
from transformer.Encoding import GeographicEncoding
from transformer.SubLayers import MultiHeadImageAttentionBlock, FeedForwardConvBlock, DownSampleBlock, UpSampleBlock


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = Encoder()
        # self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, in_channel, height, width):
        super().__init__()
        self.encoding = GeographicEncoding(seq_len, in_channel, height, width)
        self.attention_1 = MultiHeadImageAttentionBlock(3, 64, 128, in_channel, height, width)
        self.down_1 = DownSampleBlock(64, 128, 64)
        self.attention_2 = MultiHeadImageAttentionBlock(3, 128, 256, in_channel*2, height//2, width//2)

    def forward(self, x):
        x = self.encoding(x)
        return x


if '__main__' == __name__:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seq_image = torch.rand((32, 10, 64, 64, 64)).to(device)
    model = Encoder(10, 64, 64, 64).to(device)
    x = model(seq_image)
    print(x.shape)
