from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class RawTripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(RawTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs)
        inputs = inputs.view(-1, 3, inputs.size(-1))
        dist_ap = torch.pow(inputs[:, 0] - inputs[:, 1], 2).sum(dim=1)
        dist_an = torch.pow(inputs[:, 0] - inputs[:, 2], 2).sum(dim=1)
        y = torch.ones(dist_an.size(), dtype=dist_an.dtype, device=dist_an.device)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())

        dist_ap = dist[mask == 1]
        dist_ap = dist_ap.view(batch_size, -1)
        dist_an = dist[mask == 0]
        dist_an = dist_an.view(batch_size, -1)
        dist_ap, _ = torch.max(dist_ap, dim=1)
        dist_an, _ = torch.min(dist_an, dim=1)

        y = torch.ones(dist_an.size(), dtype=dist_an.dtype, device=dist_an.device)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss