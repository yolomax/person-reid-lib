from __future__ import absolute_import

from .utils import CrossEntropyLabelSmooth, CenterLoss, ContrastiveLoss, RingLoss
from .triplet import BatchHardTripletLoss, RawTripletLoss
from .oim import OIMLoss


__all__ = ['ContrastiveLoss',
           'CenterLoss',
           'CrossEntropyLabelSmooth',
           'OIMLoss',
           'RingLoss',
           'BatchHardTripletLoss',
           'RawTripletLoss']