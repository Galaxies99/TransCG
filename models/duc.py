import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseUpsamplingConvolution(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor = 2):
        super(DUC, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True)
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.layer(x)
        x = self.pixel_shuffle(x)
        return x