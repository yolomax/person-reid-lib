import torch.nn as nn
import torch.nn.functional as F
from lib.datamanager.transforms import *
from torchvision.models.inception import model_urls
from lib.network.model_factory.modelbase import ModelBase
import torch.utils.model_zoo as model_zoo
from .raw_model import Inception3
######################################################################


__all__ = ['ModelServer', 'BackboneModel']


class MyInception3(Inception3):
    def forward(self, x):
        video_flag = False
        if x.dim() == 5:
            video_flag = True
            video_num = x.size(0)
            depth = x.size(1)
            x = x.view((video_num * depth,) + x.size()[2:])

            # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            x_aux = x.view((video_num, depth) + x.size()[1:])
            x_aux = x_aux.mean(dim=1)
            aux = self.AuxLogits(x_aux)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        if video_flag:
            x = x.view((video_num, depth) + x.size()[1:])

        if self.training and self.aux_logits:
            return x, aux
        return x


class BackboneModel(nn.Module):
    def __init__(self, raw_model_dir, use_flow, logger):
        super(BackboneModel, self).__init__()
        self.use_flow = use_flow

        assert self.use_flow == False, 'The net architecture for optical flow on inception3 is not given.'
        model = MyInception3()

        model.load_state_dict(
            model_zoo.load_url(model_urls['inception_v3_google'], model_dir=raw_model_dir))
        logger.info('Model restored from pretrained inception_v3_google')

        self.fea_dim = model.fc.in_features
        self.aux_fc_in_dim = model.AuxLogits.fc.in_features
        model.fc = nn.Sequential()
        model.AuxLogits.fc = nn.Sequential()
        self.feature = model
        self.base = list(self.feature.parameters())

    def forward(self, x):
        return self.feature(x)


class ModelServer(ModelBase):
    def get_transform(self):
        if self.training:
            return [GroupResize(size=(299, 299), interpolation=3),
                    GroupToTensor(),
                    GroupNormalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                    ]
        else:
            return [GroupResize(size=(299, 299), interpolation=3),
                    GroupToTensor(),
                    GroupNormalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                    ]
