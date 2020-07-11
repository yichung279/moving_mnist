#!/usr/bin/env python3
# standard imports
import math


# thrid-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# local imports

def scaled_dot_product_attention(query, key, value):
    ''' scaled_dot_product_attention of sequntial images

    Parameters:
        query: from image_t
            shape =  (batch_size, n_head, height*width, d_k)

        key: from [image_t-1, image_t-2, ... image_t-n]
            shape =  (batch_size, n_head, n*height*width, d_k)

        value: from [image_t-1, image_t-2, ... image_t-n]
            shape =  (batch_size, n_head, n*height*width, d_v)

    Return:
        output:
            shape =  (batch_size, n_head, height*width, d_v)
        attention_head: attention matrix, means attentiom between every pixel of query and every pixel of values
            shape =  (batch_size, n_head, height*width, height*width*n)
    '''
    attention = torch.matmul(query, key.transpose(-2, -1))
    attention /= math.sqrt(query.size(-1))

    attention_head = F.softmax(attention, dim=-2)

    output = torch.matmul(attention_head, value)

    return output, attention_head


class MultiHeadImageAttentionBlock(nn.Module):
    ''' Multi-Head Attention module for images

    Use convolution layers to compute multi query matrix, key matrix and value matrix.
    After reshaping matrices into 2D matrices, compute scaled_dot_product_attention parallelly.
    Concatenate output of all scaled_dot_product_attention, and apply output convolution layer.
    Apply residaul shortcut and layer normalization,

    Parameters:
        n_head:
            number of attention layers running in parallel
        d_image:
            dimension(hidden size) of image(intput and output)
        d_k:
            dimension(hidden size) of key and query
        d_v:
            dimension(hidden size) of value
        height:
            image height
        width:
            image width
        kernel_size:
            kernel_size of convolution layers
        dropout_rate:
            dropout_rate of output convolution layer

    Forward:
    '''
    def __init__(self, n_head, d_image, d_k, d_v, height, width, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.d_image = d_image
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.height = height
        self.width = width

        self.conv_q = nn.Conv2d(d_image, d_k*n_head, kernel_size, padding=kernel_size//2)
        self.conv_k = nn.Conv2d(d_image, d_k*n_head, kernel_size, padding=kernel_size//2)
        self.conv_v = nn.Conv2d(d_image, d_v*n_head, kernel_size, padding=kernel_size//2)

        self.conv_out = nn.Conv2d(d_v*n_head, d_image, kernel_size, padding=kernel_size//2)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((d_image, height, width), eps=1e-6) #! maybe BatchNorm?


    def forward(self, query_image, seq_images):
        residaul_shortcut = query_image

        query = self.conv_q(query_image).reshape(-1, self.n_head, self.d_k, self.height*self.width)
        keys = [self.conv_k(img).reshape(-1, self.n_head, self.d_k, self.height*self.width) for img in seq_images]
        key = torch.cat(keys, dim=-1)
        values = [self.conv_v(img).reshape(-1, self.n_head, self.d_v, self.height*self.width) for img in seq_images]
        value = torch.cat(values, dim=-1)

        query, key, value = query.transpose(-2, -1), key.transpose(-2, -1), value.transpose(-2, -1)

        x, attention  = scaled_dot_product_attention(query, key, value)
        # concat
        x = x.transpose(2, 3).reshape(-1, self.n_head*self.d_v, self.height, self.width)
        x = self.conv_out(x)
        x = self.dropout(x)

        x += residaul_shortcut

        x = self.layer_norm(x + residaul_shortcut)

        return x, attention


class ConvBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel_size=3):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv2d(hidden_channel, in_channel, kernel_size, padding=kernel_size//2)

        self.layer_norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        residaul = x

        x = F.relu(self.conv_1(x))
        x = self.dropout(self.conv_2(x))

        x = self.layer_norm(x + residaul)

        return x


class DownSample(nn.Module):
    def __init__(self):
        super().__init__(in_channel, hidden_channel, kernel_size=3)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_1x1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)

        self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel_size, padding=kernel_size//2)
        self.bn_1 = nn.BatchNorm2d(in_channel)

        self.conv_2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding=kernel_size//2)
        self.bn_2 = nn.BatchNorm2d(hidden_channel)


    def forward(self, x):
        x = self.downsample(x)

        identity = self.conv_1x1(x)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        return F.relu(x + identity)


class UpSample(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel_size=3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding=kernel_size//2)
        self.bn_1 = nn.BatchNorm2d(hidden_channel)

        self.conv_2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding=kernel_size//2)
        self.bn_2 = nn.BatchNorm2d(hidden_channel)


    def forward(self, x):
        x = self.upsample(x)

        identity = x

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        return F.relu(x + identity)


if '__main__' == __name__:
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    query_image = torch.rand((32, 64, 32, 32)).to(device)
    seq_images = [torch.rand((32, 64, 32, 32)).to(device) for _ in  range(1)]

    sublayer = MultiHeadImageAttentionBlock(3, 64, 16, 12, 32, 32).to(device)
    output, attention = sublayer(query_image, seq_images)
    print(output.shape)
    print(attention.shape)
    print(attention[0])
