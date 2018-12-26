from torch.utils.data.sampler import Sampler
import numpy as np


class SamplerBase(Sampler):
    def __init__(self, data_info, instance_num, sample_num, sample_len, npr):
        super().__init__(data_info)
        self.npr = npr

        self.data_info = data_info.copy()
        self.person_num = np.unique(self.data_info[:, 0]).size
        self.frames_len_min = np.min(self.data_info[:, 4])
        self.frames_len_max = np.max(self.data_info[:, 4])

        self.instance_num = instance_num
        self.sample_num = sample_num
        self.sample_len = sample_len

        if self.frames_len_max == 1 and self.frames_len_min == 1:
            self._process_raw_data = self._process_raw_image_data
        else:
            self._process_raw_data = self._process_raw_video_data

        self.init()

    def init(self):
        raise NotImplementedError

    def _process_raw_image_data(self, tracklet):
        output = []
        for track_i in tracklet:
            track_info = [track_i[0], track_i[1], track_i[2]]  # person id, cam_id, img_idx
            output.append(track_info)
        return output

    def _process_raw_video_data(self, tracklet):
        output = []
        for track_i in tracklet:
            track_info = [track_i[0], track_i[1]]  # person id, cam_id

            track_i_begin = track_i[2]
            track_i_raw_len = track_i[4]
            track_i_sample_len = min(track_i_raw_len, self.sample_len)
            track_i_start_id = self.npr.random_integers(0, track_i_raw_len - track_i_sample_len)
            track_i_sample_begin = track_i_begin + track_i_start_id
            track_i_sample_end = track_i_begin + track_i_start_id + track_i_sample_len
            track_info.extend(list(range(track_i_sample_begin, track_i_sample_end)))
            output.append(track_info)
        return output

    def __len__(self):
        return self._len



