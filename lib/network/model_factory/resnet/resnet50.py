import torch.nn as nn
import torch.nn.functional as F
from lib.datamanager.transforms import *
from torchvision.models.resnet import Bottleneck, ResNet, model_urls
from lib.network.model_factory.modelbase import ModelBase
import torch.utils.model_zoo as model_zoo
######################################################################


__all__ = ['ModelServer', 'BackboneModel']


class BackboneModel(nn.Module):
    def __init__(self, raw_model_dir, use_flow, logger):
        super(BackboneModel, self).__init__()
        self.use_flow = use_flow
        model = ResNet(Bottleneck, [3, 4, 6, 3])

        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet50'], model_dir=raw_model_dir))
        logger.info('Model restored from pretrained resnet50')

        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.base = list(self.feature.parameters())

        if self.use_flow:
            self.flow_branch = self.get_flow_branch(model)
            self.rgb_branch = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.fuse_branch = nn.Sequential(*list(model.children())[4:-2])
        self.fea_dim = model.fc.in_features

    def get_flow_branch(self, base_model):
        conv_layer = base_model.conv1

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        conv_flow = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)

        conv_flow.weight.data = new_kernels
        if len(params) == 2:
            conv_flow.bias.data = params[1].data  # add bias if neccessary

        bn_flow = nn.BatchNorm2d(conv_layer.out_channels)
        return nn.Sequential(conv_flow, bn_flow, base_model.relu, base_model.maxpool)

    def rgb_flow_forward(self, x):
        rgb = x[0]
        flow = x[1]

        video_flag = False
        if rgb.dim() == 5:
            video_flag = True
            video_num = rgb.size(0)
            depth = rgb.size(1)
            rgb = rgb.view((video_num * depth,) + rgb.size()[2:])
            flow = flow.view((video_num * depth,) + flow.size()[2:])
        rgb = self.rgb_branch(rgb)
        flow = self.flow_branch(flow)
        x = self.fuse_branch(rgb + flow)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if video_flag:
            x = x.view((video_num, depth) + x.size()[1:])

        return x

    def rgb_forward(self, x):
        video_flag = False
        if x.dim() == 5:
            video_flag = True
            video_num = x.size(0)
            depth = x.size(1)
            x = x.view((video_num * depth,) + x.size()[2:])

        x = self.feature(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if video_flag:
            x = x.view((video_num, depth) + x.size()[1:])

        return x

    def forward(self, x):
        if not self.use_flow:
            return self.rgb_forward(x)
        else:
            return self.rgb_flow_forward(x)


class ModelServer(ModelBase):
    pass
