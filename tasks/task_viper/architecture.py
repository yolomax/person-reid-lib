import torch.nn as nn
from lib.network.model_factory.resnet import ModelServer, NetServer, BackboneModel
from lib.network.loss_factory import BatchHardTripletLoss


class FuseNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = input_size
        self.H = 8
        self.W = 4
        self.HW = self.H * self.W

    def forward(self, x, is_training=True):
        if x.dim() == 2:
            x = x.unsqueeze(dim=0)
        assert x.dim() == 3
        x = x.mean(dim=1)
        return x


class ModelClient(ModelServer):
    def __init__(self, num_classes, num_camera, use_flow, is_image_dataset, raw_model_dir, logger):
        super().__init__(use_flow, is_image_dataset, logger)

        model = self.get_model(BackboneModel, raw_model_dir, logger)
        self.backbone_fea_dim = model.fea_dim
        self.fea_dim = self.backbone_fea_dim
        self.net_info = ['backbone feature dim: ' + str(self.backbone_fea_dim)]
        self.net_info.append('final feature dim: ' + str(self.fea_dim))
        self.base = model.base

        if not self.is_image_dataset:
            self.fuse_net = FuseNet(self.backbone_fea_dim, self.fea_dim)
        self.classifier = self.get_classifier(self.fea_dim, num_classes)
        self.feature = model

        self.distance_func = 'L2Euclidean'

    def forward(self, x):
        fea = self.feature(x)
        if not self.is_image_dataset:
            fea = self.fuse_net(fea)
        logits = self.classifier(fea)
        return fea, logits


class NetClient(NetServer):
    def init_options(self):
        self.contrast = BatchHardTripletLoss(margin=0.3)
        self.line_name = ['Identity', 'Triplet',
                          'All']

    def compute_loss(self, model_output, label_identity):
        fea, logits_i = model_output

        loss_identity_i = self.identity(logits_i, label_identity)
        loss_v = self.contrast(fea, label_identity)
        loss_final = loss_identity_i + loss_v

        self.loss_mean.updata([loss_identity_i.item(),
                               loss_v.item(),
                               loss_final.item()])
        return loss_final
