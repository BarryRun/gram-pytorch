import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    # @torchsnooper.snoop()
    def forward(self, y_hat, y, length):
        # calculate the loss
        logEps = torch.tensor(1e-8)
        tmp1 = y * torch.log(y_hat + logEps)
        tmp2 = (1. - y) * torch.log(1. - y_hat + logEps)
        cross_entropy = -(tmp1 + tmp2)
        tmp = torch.sum(cross_entropy, 2)
        tmp = torch.sum(tmp, 0)
        log_likelyhood = torch.div(tmp, length)
        cost = torch.mean(log_likelyhood)

        # calculate the Accuracy@20
        y_sorted, indices = torch.sort(y_hat, dim=2, descending=True)
        TP = 0.
        TN = 0.
        indices = indices[:, :, :20]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            if k in indices[i][j]:
                TP += 1
            else:
                TN += 1
        acc = TP / (TP + TN + 1)
        return cost, acc
