from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomCrop
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
import numpy as np
import random
import numbers
import torch
import math


__all__ = ['GroupToTensor',
           'GroupResize',
           'GroupRandomHorizontalFlip',
           'GroupRandomCrop',
           'GroupToTensor',
           'GroupNormalize',
           'StackTensor',
           'GroupToPILImage',
           'GroupRandom2DTranslation',
           'GroupRandomErasing']


class ImageData(object):
    def __init__(self, img, x=None, y=None):
        self.img = img
        self.x = x
        self.y = y


def _group_process(images, func, params):
    if isinstance(images, (tuple, list)):
        return [_group_process(img, func, params) for img in images]
    else:
        return func(images, params)


class GroupOperation(object):
    def _instance_process(self, images, params):
        raise NotImplementedError

    def _get_params(self, images):
        return None

    def __call__(self, images):
        params = self._get_params(images)
        return _group_process(images, self._instance_process, params)


class GroupToPILImage(GroupOperation, ToPILImage):
    def __init__(self, mode=None, use_flow=False):
        super(GroupToPILImage, self).__init__(mode)
        self.use_flow = use_flow

    def _instance_process(self, pic_list, params):
        if isinstance(pic_list, np.ndarray):
            if pic_list.ndim == 3:
                return self.to_pil_image(pic_list)
            elif pic_list.ndim == 4:
                return [self.to_pil_image(pic_i) for pic_i in range(pic_list.shape[0])]
            else:
                raise TypeError
        raise TypeError

    def to_pil_image(self, pic):
        if pic.shape[2] == 3:
            return ImageData(F.to_pil_image(pic, self.mode))
        elif pic.shape[2] == 1:
            return ImageData(F.to_pil_image(pic))
        elif pic.shape[2] == 5:
            if self.use_flow:
                pic_rgb = F.to_pil_image(pic[..., :3], self.mode)
                pic_x = F.to_pil_image(pic[..., 3:4])
                pic_y = F.to_pil_image(pic[..., 4:5])
                return ImageData(pic_rgb, pic_x, pic_y)
            else:
                return ImageData(F.to_pil_image(pic[..., :3], self.mode))
        else:
            raise ValueError


class GroupResize(GroupOperation, Resize):
    def _instance_process(self, img, params):
        img.img = F.resize(img.img, self.size, self.interpolation)
        if img.x is not None:
            img.x = F.resize(img.x, self.size, self.interpolation)
        if img.y is not None:
            img.y = F.resize(img.y, self.size, self.interpolation)

        return img


class GroupRandomHorizontalFlip(GroupOperation, RandomHorizontalFlip):
    def _get_params(self, images):
        if random.random() < self.p:
            return True
        else:
            return False

    def _instance_process(self, img, flip_flag):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if flip_flag:
            img.img = F.hflip(img.img)
            if img.x is not None:
                img.x = ImageOps.invert(img.x)
        return img


class GroupRandomCrop(GroupOperation):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def pad_func(self, img, params):
        if self.padding is not None:
            img.img = F.pad(img.img, self.padding, self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, self.padding, self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.img.size[0] < self.size[1]:
            img.img = F.pad(img.img, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)

        if self.pad_if_needed and img.img.size[1] < self.size[0]:
            img.img = F.pad(img.img, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)

        return img

    def _get_params(self, images):
        """
        Args:
            img (PIL Image) list: Image to be cropped.
        Returns:
            PIL Image list: Cropped image.
        """
        while isinstance(images, (tuple, list)):
            images = images[0]
        img = images.img

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return self.get_params(img, self.size)

    def _instance_process(self, images, params):
        i, j, h, w = params
        img = _group_process(images, self.pad_func, None)
        img.img = F.crop(img.img, i, j, h, w)

        if img.x is not None:
            img.x = F.crop(img.x, i, j, h, w)
        if img.y is not None:
            img.y = F.crop(img.y, i, j, h, w)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class GroupToTensor(GroupOperation, ToTensor):
    def _instance_process(self, img, params):
        img.img = F.to_tensor(img.img)
        if img.x is not None:
            img.x = F.to_tensor(img.x)
        if img.y is not None:
            img.y = F.to_tensor(img.y)

        return img


class GroupNormalize(GroupOperation, Normalize):
    def _instance_process(self, image, params):
        image.img = F.normalize(image.img, self.mean[:3], self.std[:3])
        if image.x is not None:
            image.x = F.normalize(image.x, self.mean[3:4], self.std[3:4])
        if image.y is not None:
            image.y = F.normalize(image.y, self.mean[3:4], self.std[3:4])
        return image


class GroupRandom2DTranslation(GroupOperation):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def _get_params(self, images):
        if random.uniform(0, 1) > self.p:
            return None
        else:
            new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
            x_maxrange = new_width - self.width
            y_maxrange = new_height - self.height
            x1 = int(round(random.uniform(0, x_maxrange)))
            y1 = int(round(random.uniform(0, y_maxrange)))
            return new_width, new_height, x1, y1

    def _instance_process(self, img, params):
        if params is None:
            img.img = img.img.resize((self.width, self.height), self.interpolation)

            if img.x is not None:
                img.x = img.x.resize((self.width, self.height), self.interpolation)
            if img.y is not None:
                img.y = img.y.resize((self.width, self.height), self.interpolation)

        else:
            new_width, new_height, x1, y1 = params
            img.img = img.img.resize((new_width, new_height), self.interpolation)
            img.img = img.img.crop((x1, y1, x1 + self.width, y1 + self.height))

            if img.x is not None:
                img.x = img.x.resize((new_width, new_height), self.interpolation)
                img.x = img.x.crop((x1, y1, x1 + self.width, y1 + self.height))

            if img.y is not None:
                img.y = img.y.resize((new_width, new_height), self.interpolation)
                img.y = img.y.crop((x1, y1, x1 + self.width, y1 + self.height))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GroupRandomErasing(GroupOperation):
    """ Randomly selects a rectangle region in an image and erases its pixels.
            'Random Erasing Data Augmentation' by Zhong et al.
            See https://arxiv.org/pdf/1708.04896.pdf
        Args:
             probability: The probability that the Random Erasing operation will be performed.
             sl: Minimum proportion of erased area against input image.
             sh: Maximum proportion of erased area against input image.
             r1: Minimum aspect ratio of erased area.
             mean: Erasing value.
        """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def _instance_process(self, img, params):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.img.size()[1] * img.img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.img.size()[2] and h < img.img.size()[1]:
                x1 = random.randint(0, img.img.size()[1] - h)
                y1 = random.randint(0, img.img.size()[2] - w)
                if img.img.size()[0] == 3:
                    img.img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img.img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img.img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img.img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                if img.x is not None:
                    img.x[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                if img.y is not None:
                    img.y[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class StackTensor(object):
    def __call__(self, tensor_list):
        if isinstance(tensor_list, (tuple, list)):
            rgb_tensor = []
            flow_tensor = []
            for tensor_i in tensor_list:
                rgb_tensor.append(tensor_i.img)
                if tensor_i.x is not None and tensor_i.y is not None:
                    flow_tensor.append(torch.cat([tensor_i.x, tensor_i.y], dim=0))
            if len(tensor_list) > 1:
                rgb_tensor = torch.stack(rgb_tensor)

                if len(flow_tensor) > 1:
                    flow_tensor = torch.stack(flow_tensor)
                    return rgb_tensor, flow_tensor

                return rgb_tensor
            else:
                if len(flow_tensor) > 0:
                    return rgb_tensor[0], flow_tensor[0]
                return rgb_tensor[0]
        raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToSpaceBGR(object):

    def __init__(self, is_bgr=True):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255=True):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor