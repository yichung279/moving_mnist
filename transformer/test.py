#!/usr/bin/env python3
import torch
import torch.nn as nn

if '__main__' == __name__:
    input = torch.randn(20, 16, 50, 100)
    # exact output size can be also specified as an argument
    input = torch.randn(1, 16, 32, 32)
    upsample = nn.ConvTranspose2d(16, 16, 2, stride=2)
    output = upsample(input)
    print(output.size())

