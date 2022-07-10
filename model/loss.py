import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class MutilLabelSmoothing(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.alpha = alpha

    def forward(self, pred, target):

        B, D = target.shape
        tag_num = torch.sum(target, dim=-1)
        Smoothing = tag_num * (1 - self.alpha) / D
        target = target * self.alpha + Smoothing.reshape((B, 1))
        loss = self.bce(pred, target)
        return loss


class DistMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, scores):
        dist = pred - scores
        b = dist.shape
        b = b[0]
        dist_j = dist[1:]
        dist_i = dist[:b - 1]
        loss = self.mse(pred, scores) + torch.sum(torch.pow(dist_j - dist_i, 2)) / (b - 1)
        return loss


class CrossMSE(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, scores):
        dist = pred - scores
        losses = 2 * (self.mse(pred, scores) - torch.pow(torch.mean(dist), 2))
        return losses


class FusionCrossMSE(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, scores):
        dist = pred - scores
        losses = 2 * self.mse(pred, scores) - torch.pow(torch.mean(dist), 2)
        return losses