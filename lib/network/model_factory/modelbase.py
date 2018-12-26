import torch.nn as nn
import torch.nn.functional as F
from lib.datamanager.transforms import *
from lib.network.layer_factory.utils import weights_init_classifier, weights_init_kaiming
######################################################################


__all__ = ['ModelBase']


class ModelBase(nn.Module):
    def __init__(self, use_flow, is_image_dataset, logger):
        super(ModelBase, self).__init__()
        self.is_image_dataset = is_image_dataset
        if not self.is_image_dataset and use_flow:
            self.use_flow = True
            logger.info('Optical flow is used.')
        else:
            self.use_flow = False

    def get_classifier(self, hidden_size, num_classes):
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(hidden_size, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  # default dropout rate 0.5
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        fc = add_block

        classifier = []
        classifier += [nn.Linear(num_bottleneck, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        classifier = nn.Sequential(fc, classifier)
        return classifier

    def fea_process_func(self, fea):
        '''

        :param fea: shape [N, fea_shape], N is the batch size, fea_shape is the feature shape of each image or video
        :return:
        '''
        if self.is_image_dataset:
            return fea.detach()
        else:
            output = self.fuse_net(fea, is_training=False)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.view(-1)

            return output.detach()

    def get_transform(self):
        if self.training:
            return [GroupRandom2DTranslation(256, 128),
                    GroupRandomHorizontalFlip(),
                    GroupToTensor(),
                    GroupNormalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.226])
                    ]
        else:
            return [GroupResize(size=(256, 128), interpolation=3), #Image.BICUBIC
                    GroupToTensor(),
                    GroupNormalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.226])
                    ]

    def get_model(self, BackboneModel, raw_model_dir, logger):
        model = BackboneModel(raw_model_dir, self.use_flow, logger)
        return model
