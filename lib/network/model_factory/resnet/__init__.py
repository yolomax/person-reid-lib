from torch import optim
import torch.nn as nn
from lib.network.model_factory.netbase import NetBase
from lib.network.model_factory.resnet.resnet50 import ModelServer, BackboneModel

__all__ = ['ModelServer', 'NetServer', 'BackboneModel']


class NetServer(NetBase):
    def init_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        base_params = list(map(id, self.model.base))

        new_params = filter(lambda p: id(p) not in base_params, self.model.parameters())

        self.info(list(new_params))

        return optimizer

    def const_options(self):
        self.lr = 0.0003
        self.weight_decay = 5e-4
        self.lr_decay_step = [20, 40]
        self.gamma = 0.1

        self.identity = nn.CrossEntropyLoss().cuda()

    def init_options(self):
        self.line_name = ['Identity', 'All']