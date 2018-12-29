from torch import optim
import torch.nn as nn
from lib.network.model_factory.netbase import NetBase
from .inception_v3 import ModelServer, BackboneModel

__all__ = ['ModelServer', 'NetServer', 'BackboneModel']


class NetServer(NetBase):
    def init_optimizer(self):
        base_params = list(map(id, self.model.base))

        new_params = filter(lambda p: id(p) not in base_params, self.model.parameters())

        self.info(list(new_params))

        new_params = filter(lambda p: id(p) not in base_params, self.model.parameters())

        # Observe that all parameters are being optimized
        optimizer = optim.SGD([
            {'params': self.model.base, 'lr': 0.1 * self.lr},
            {'params': new_params, 'lr': self.lr},
        ], momentum=0.9, weight_decay=self.weight_decay, nesterov=True)

        return optimizer

    def const_options(self):
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.lr_decay_step = [40000]
        self.gamma = 0.1

        self.identity = nn.CrossEntropyLoss().cuda()

    def init_options(self):
        self.line_name = ['Identity', 'All']