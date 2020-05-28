#!/usr/bin/env python3
# standard imports
import math


# thrid-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# local imports

def scaled_dot_product_attention(query, key, value):
    ''' scaled_dot_product_attention of sequmtial images

    param:
        query: from image_t
            shape =  batch_size, n_head, height*width, in_channel

        key: from [image_t-1, image_t-2, ... image_t-n]
            shape =  batch_size, n_head, n*height*width, in_channel

        value: from [image_t-1, image_t-2, ... image_t-n]
            shape =  batch_size, n_head, n*height*width, out_channel

    return:
        output:
            shape =  batch_size, n_head, height*width, out_channel
        attention_head: attention matrix, means attentiom between every pixel of query and every pixel of values
            shape =  batch_size, n_head, height*width, height*width*n
    '''
    attention = torch.matmul(query, key.transpose(-2, -1))
    attention /= math.sqrt(query.size(-1))

    attention_head = F.softmax(attention, dim=-2)

    output = torch.matmul(attention_head, value)
    # output = output.reshape((query.size(0), self.n_head, height, weight, self.channel)

    return output, attention_head


class multiHeadImageAttentionBlock(nn.Module):
    def __init__(self, n_head, channel, height, width, d_value, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.channel = channel
        self.height = height
        self.width = width
        self.d_value = d_value

        self.conv_q = nn.Conv2d(channel, channel*n_head, kernel_size, padding=kernel_size//2)
        self.conv_k = nn.Conv2d(channel, channel*n_head, kernel_size, padding=kernel_size//2)
        self.conv_v = nn.Conv2d(channel, d_value*n_head, kernel_size, padding=kernel_size//2)

        self.conv_out = nn.Conv2d(self.n_head*self.d_value, channel, kernel_size, padding=kernel_size//2)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((channel, height, width), eps=1e-6) #! maybe BatchNorm?


    def forward(self, query_image, seq_images):
        residaul_shortcut = query_image

        query = self.conv_q(query_image).reshape(-1, self.n_head, self.channel, self.height*self.width)
        keys = [self.conv_k(img).reshape(-1, self.n_head, self.channel, self.height*self.width) for img in seq_images]
        key = torch.cat(keys, dim=-1)
        values = [self.conv_v(img).reshape(-1, self.n_head, self.d_value, self.height*self.width) for img in seq_images]
        value = torch.cat(values, dim=-1)

        query, key, value = query.transpose(-2, -1), key.transpose(-2, -1), value.transpose(-2, -1)

        x, attention  = scaled_dot_product_attention(query, key, value)
        # concat
        x = x.transpose(2, 3).reshape(-1, self.n_head*self.d_value, self.height, self.width)
        x = self.conv_out(x)

        x += residaul_shortcut

        x = self.layer_norm(x + residaul_shortcut)

        return x, attention



if '__main__' == __name__:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # query = torch.rand((32, 3, 16*16, 64)).to(device)
    # key = torch.rand((32, 3, 10*16*16, 64)).to(device)
    # value = torch.rand((32, 3, 10*16*16, 128)).to(device)
    #
    # output, att = scaled_dot_product_attention(query, key, value)
    # print(output.shape, att.shape)


    query_image = torch.rand((32, 64, 32, 32)).to(device)
    seq_images = [torch.rand((32, 64, 32, 32)).to(device) for _ in  range(10)]

    # multiHeadImageAttentionBlock( n_head, channel, height, width, d_value, kernel_size=3, dropout_rate=0.1):
    sublayer = multiHeadImageAttentionBlock(3, 64, 32, 32, 128).to(device)
    output, attention = sublayer(query_image, seq_images)
    print(output.shape)
    print(attention.shape)
    print(attention[0])
