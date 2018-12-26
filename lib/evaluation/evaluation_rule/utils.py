from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import numpy as np


class TensorBuffer(object):
    def __init__(self, block_size_list, func):
        self.block_size_list = block_size_list
        self.block_size_max = max(self.block_size_list)
        self._func = func

        if self.block_size_max == 1:
            self.push = self.push_image
        else:
            self.push = self.push_video
            self.idx = 0
            self.fea_shape = None

        self.result = []
        self.block_idx = 0
        self.is_end = False

    def push_image(self, fea):
        batch_size = fea.size(0)
        for i_fea in range(batch_size):
            self.result.append(fea[i_fea])
            self.block_idx += 1
            if self.block_idx == len(self.block_size_list):
                self.is_end = True

    def push_video(self, fea):
        fea_shape = fea.size()
        batch_size = fea_shape[0]
        if self.fea_shape is None:
            fea_shape = fea_shape[1:]
            self.fea_shape = fea_shape
            self.feaMat = torch.from_numpy(np.zeros((self.block_size_max,) + fea_shape, np.float32)).cuda()

        read_range = [0, 0]
        store_range = [self.idx, self.idx]
        for i_batch in range(batch_size):
            read_range[1] += 1
            store_range[1] += 1
            self.idx += 1
            if self.idx == self.block_size_list[self.block_idx]:
                self.feaMat[store_range[0]:store_range[1]] = fea[read_range[0]:read_range[1], ...]
                self.result.append(self._func(self.feaMat[:self.block_size_list[self.block_idx]]))

                self.idx = 0
                read_range[0] = read_range[1]
                store_range = [0, 0]
                self.block_idx += 1
                if self.block_idx == len(self.block_size_list):
                    self.is_end = True

        if read_range[0] != read_range[1]:
            self.feaMat[store_range[0]:store_range[1]] = fea[read_range[0]:read_range[1], ...]