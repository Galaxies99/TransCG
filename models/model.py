import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthFillerNet(nn.Module):
    def __init__(self):
        super(DepthFillerNet, self).__init__()
    