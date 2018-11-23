import torch.nn.functional as F
import numpy as np
from lib.utils.util import ConstType


class Distance(ConstType):
    def __init__(self, func):
        self._distance_algorithm = {'Euclidean': self._euclidean,
                                    'L2Euclidean': self._l2_euclidean,
                                    'Cosine': self._cosine
                                    }
        if isinstance(func, str):
            if func not in self._distance_algorithm:
                print('No this distance func ' + func)
                raise KeyError
            self._algorithm = self._distance_algorithm[func]
        else:
            self._algorithm = func

    @staticmethod
    def _l2_euclidean(vect_p, vect_g):
        assert vect_p.dim() == 2 and vect_g.dim() == 2
        vect_p = F.normalize(vect_p)
        vect_g = F.normalize(vect_g)
        dst = F.pairwise_distance(vect_p, vect_g)
        return dst

    @staticmethod
    def _euclidean_np(vect_left, vect_right):
        assert vect_left.ndim == 1 and vect_right.ndim == 1
        return np.sqrt(np.sum(np.square(vect_left - vect_right)))

    @staticmethod
    def _cosine_np(vect_left, vect_right):
        assert vect_left.ndim == 1 and vect_right.ndim == 1
        return 1 - np.sum((vect_left * vect_right) / np.linalg.norm(vect_left) / np.linalg.norm(vect_right))

    @staticmethod
    def _euclidean(vect_p, vect_g):
        dst = F.pairwise_distance(vect_p, vect_g)
        return dst

    @staticmethod
    def _cosine(vect_p, vect_g):
        dst = 1 - F.cosine_similarity(vect_p, vect_g)
        return dst

    def __call__(self, *args, **kwargs):
        return self._algorithm(*args, **kwargs)