import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_loss():
    loss_temp = torch.nn.CrossEntropyLoss()
    return loss_temp

def kl_loss():
    loss_temp = torch.nn.KLDivLoss()
    return loss_temp

def dist_loss(loss_soft,loss_hard,T,alpha):
    loss_soft2 = loss_soft *T *T
    loss_temp = (1-alpha)*loss_hard + alpha *loss_soft2
 
    return loss_temp

class PartitionLoss():
    def __init__(self, device):
        super(PartitionLoss, self).__init__()
        self.fhead_loss  = HeadLoss()

    def loss(self,feat,targets,heads):    
        loss = self.fhead_loss(heads)
        return loss


class HeadLoss(nn.Module):
    def __init__(self, ):
        super(HeadLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1+num_head/var)
        else:
            loss = 0
            
        return loss

class FeatLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(FeatLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
