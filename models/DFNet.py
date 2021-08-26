import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution


class DFNet(nn.Module):
    def __init__(self, in_channels = 4, hidden_channels = 64, L = 5, k = 12, **kwargs):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense4_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense1_duc = DenseUpsamplingConvolution(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense2_duc = DenseUpsamplingConvolution(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense3_duc = DenseUpsamplingConvolution(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense4_duc = DenseUpsamplingConvolution(self.k, self.hidden_channels, upscale_factor = 2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(True)
        )
    
    def forward(self, rgb, depth):
        # 720 x 1280 (rgb, depth) -> 360 x 640 (h)
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)
        h = self.first(torch.cat((rgb, depth), dim = 1))

        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(depth, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        h_d1s = self.dense1s(torch.cat((h_d1s, depth1), dim = 1))
        h_d1s = self.dense1s_conv2(h_d1s)
        # dense1: normal
        h = self.dense1_conv1(h)
        h = self.dense1(torch.cat((h, depth1), dim = 1))
        h = self.dense1_conv2(h)

        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)
        h_d2s = self.dense2s(torch.cat((h_d2s, depth2), dim = 1))
        h_d2s = self.dense2s_conv2(h_d2s)
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.dense2(torch.cat((h, depth2), dim = 1))
        h = self.dense2_conv2(h)
        
        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim = 1))
        h_d3s = self.dense3s_conv2(h_d3s)
        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.dense3(torch.cat((h, depth3), dim = 1))
        h = self.dense3_conv2(h)

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        h = self.dense4_conv1(h)
        h = self.dense4(torch.cat((h, depth4), dim = 1))
        h = self.dense4_conv2(h)

        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(h)
        h = self.updense1(torch.cat((h, depth4), dim = 1))
        h = self.updense1_duc(h)

        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.updense2_conv(h)
        h = self.updense2(torch.cat((h, depth3), dim = 1))
        h = self.updense2_duc(h)

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.updense3_conv(h)
        h = self.updense3(torch.cat((h, depth2), dim = 1))
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.updense4_conv(h)
        h = self.updense4(torch.cat((h, depth1), dim = 1))
        h = self.updense4_duc(h)

        # final
        h = self.final(h)

        return rearrange(h, 'n 1 h w -> n h w')

