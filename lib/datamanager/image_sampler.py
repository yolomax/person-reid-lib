import numpy as np
from torch.utils.data.sampler import Sampler


class ImageSampler(Sampler):
    def __init__(self, dataset, iter_size=10000, batch_size=32, npr=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.train_track = self.dataset.train_track
        self.npr = npr
        self.track_num = self.train_track.shape[0]
        self.person_num = self.dataset.train_person_num
        self.frames_len_min = np.min(self.train_track[:, 4])
        self.frames_len_max = np.max(self.train_track[:, 4])
        self.iter_size = iter_size
        self.batch_size = batch_size
        self.num = self.dataset.train_image_num
        self.data = self.get_label()

        self.idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter_size

    def get_label(self):
        data = np.zeros((self.num, 2), np.int64)
        idx = 0
        for i_track in range(self.track_num):
            label_i = self.train_track[i_track, 0]
            for image_i in range(self.train_track[i_track, 2], self.train_track[i_track, 3]):
                data[idx, 0] = label_i
                data[idx, 1] = image_i
                idx += 1
        return data

    def __next__(self):
        if self.idx >= self.iter_size:
            raise StopIteration
        tmp_idx = self.npr.choice(self.num, self.batch_size, False)
        output = self.data[tmp_idx].tolist()
        self.idx += 1
        return output