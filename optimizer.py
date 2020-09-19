import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    @torchsnooper.snoop()
    def forward(self, y_hat, y):
        logEps = torch.tensor(1e-8)
        cross_entropy = -(y * torch.log(y_hat + logEps) + (1. - y) * torch.log(1. - y_hat + logEps))
        tmp = torch.sum(cross_entropy, 2)
        tmp = torch.sum(tmp, 0)

