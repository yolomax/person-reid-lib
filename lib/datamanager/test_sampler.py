import numpy as np
from torch.utils.data.sampler import Sampler


class TestSampler(Sampler):
    def __init__(self, data_info, batch_size):
        super().__init__(data_info)
        self.test_info = data_info.copy()
        self.track_num = self.test_info.shape[0]
        self.image_num = np.sum(self.test_info[:, -1])
        self.batch_size_max = batch_size

        self.image_member = self._get_image_member()
        self.iter_num = (self.image_num + self.batch_size_max - 1) // self.batch_size_max
        self.idx = 0

    def _get_image_member(self):
        data = np.zeros((self.image_num, 3), dtype=np.int64)
        idx = 0
        for i_track in range(self.track_num):
            track_i = self.test_info[i_track]
            for frames_i in range(track_i[2], track_i[3]):
                data[idx, 0] = track_i[0]
                data[idx, 1] = track_i[1]
                data[idx, 2] = frames_i
                idx += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter_num

    def __next__(self):
        if self.idx >= self.image_num:
            raise StopIteration
        if self.idx + self.batch_size_max >= self.image_num:
            output = self.image_member[self.idx:].tolist()
            self.idx = self.image_num
        else:
            output = self.image_member[self.idx:self.idx+self.batch_size_max].tolist()
            self.idx += self.batch_size_max
        return output
