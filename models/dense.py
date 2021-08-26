import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import xavier_init


class DenseBlock(nn.Module):
    def __init__(self, in_channels, L, k, with_bn = False):
        '''
        Dense Block.
        '''
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.L = L
        self.k = k
        self.with_bn = with_bn
        self.layers = []
        for i in range(L):
            layer_in_channels = in_channels + i * k + 1
            single_layer = []
            conv1 = nn.Conv2d(layer_in_channels, k * 4, kernel_size = 1, stride = 1)
            xavier_init(conv1)
            single_layer.append(conv1)
            if with_bn:
                single_layer.append(nn.BatchNorm2d(k * 4))
            single_layer.append(nn.ReLU(True))
            conv2 = nn.Conv2d(k * 4, k, kernel_size = 3, stride = 1, padding = 1)
            xavier_init(conv2)
            single_layer.append(conv2)
            if with_bn:
                single_layer.append(nn.BatchNorm2d(k))
            single_layer.append(nn.ReLU(True))
            self.layers.append(nn.Sequential(*single_layer))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        h = x
        hs = [h]
        for i in range(self.L):
            if i != 0:
                h = torch.cat(hs, dim = 1)
            h = self.layers[i](h)
            if i != self.L - 1:
                hs.append(h)
        return h
