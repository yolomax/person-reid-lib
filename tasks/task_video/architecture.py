import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.network.model_factory.inception_v3 import ModelServer, NetServer, BackboneModel
from lib.network.loss_factory import BatchHardTripletLoss
from lib.network.layer_factory.utils import BasicConv2d


class FuseNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        bottleneck_size = [256, 128]
        self.reduce_dim_z = BasicConv2d(input_size * 2, bottleneck_size[0], kernel_size=1, padding=0)
        self.s_atten_z = nn.Sequential(
            nn.Conv2d(1, bottleneck_size[1], kernel_size=8, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_size[1], 64, kernel_size=1, padding=0, bias=False))
        self.c_atten_z = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(bottleneck_size[0], input_size, kernel_size=1, padding=0, bias=False))

        self.t_info = nn.Sequential(
            nn.Conv3d(input_size, 256, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, self.hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm3d(self.hidden_size),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(8)

    def generate_attention_z(self, x):
        z = self.reduce_dim_z(x)
        atten_s = self.s_atten_z(z.mean(dim=1, keepdim=True)).view(z.size(0), 1, z.size(2), z.size(3))
        atten_c = self.c_atten_z(z)
        z = F.sigmoid(atten_s * atten_c)
        return z, 1 - z

    def forward(self, x, is_training=True):
        if x.dim() == 4:
            x = x.view((1,) + x.size())
        assert x.dim() == 5

        video_num = x.size(0)
        depth = x.size(1)

        res = torch.cat((x[:, 0].contiguous().view((x.size(0), 1) + x.size()[2:]), x), dim=1)
        res = res[:, :-1]
        res = x - res

        h = x[:, 0]
        output = []
        for t in range(depth):
            con_fea = torch.cat((h - x[:, t], res[:, t]), dim=1)
            z_p, z_r = self.generate_attention_z(con_fea)
            h = z_r * h + z_p * x[:, t]
            output.append(h)

        fea_t = torch.stack(output, dim=2)
        fea_s = fea_t.mean(dim=4).mean(dim=2)
        fea_t = self.pool(self.t_info(fea_t).mean(dim=2)).view(video_num, -1)
        return fea_t, fea_s


class ModelClient(ModelServer):
    def __init__(self, num_classes, num_camera, use_flow, is_image_dataset, raw_model_dir, logger):
        super().__init__(use_flow, is_image_dataset, logger)

        model = self.get_model(BackboneModel, raw_model_dir, logger)
        self.backbone_fea_dim = model.fea_dim
        self.fea_dim = 256
        aux_fc_in_dim = model.aux_fc_in_dim
        self.net_info = ['backbone feature dim: ' + str(self.backbone_fea_dim)]
        self.net_info.append('final feature dim: ' + str(self.fea_dim))
        self.base = model.base
        model.feature.AuxLogits.fc = nn.Linear(aux_fc_in_dim, num_classes)
        model.feature.AuxLogits.fc.stddev = 0.001

        if not self.is_image_dataset:
            self.fuse_net = FuseNet(self.backbone_fea_dim, self.fea_dim)
        self.classifier = self.get_classifier(self.fea_dim, num_classes)
        self.feature = model

        self.distance_func = 'L2Euclidean'

    def forward(self, x):
        fea, aux = self.feature(x)
        fea_t, fea_s_p = self.fuse_net(fea)

        logits = self.classifier(fea_t)
        return fea_s_p, fea_t, logits, aux


class NetClient(NetServer):
    def init_options(self):
        self.contrast = BatchHardTripletLoss(margin=0.4)
        self.line_name = ['Identity',
                          'Aux',
                          'Part',
                          'Video', 'All']

    def get_part_loss(self, fea_part, target):
        h_dim = fea_part.size(2)
        loss_part = self.contrast(fea_part[:, :, 0], target)
        for part_i in range(1, h_dim):
            loss_part += self.contrast(fea_part[:, :, part_i], target)
        return 1.0 / h_dim * loss_part

    def compute_loss(self, model_output, label_identity):
        fea_s_p, fea_t_v, logits_i, logits_a = model_output

        loss_identity_i = self.identity(logits_i, label_identity)
        loss_identity_a = self.identity(logits_a, label_identity)

        loss_s_p = self.get_part_loss(fea_s_p, label_identity)
        loss_t_v = self.contrast(fea_t_v, label_identity)

        loss_final = loss_identity_i + loss_identity_a + loss_t_v + loss_s_p

        self.loss_mean.updata([loss_identity_i.item(),
                               loss_identity_a.item(),
                               loss_s_p.item(),
                               loss_t_v.item(),
                               loss_final.item()])
        return loss_final
