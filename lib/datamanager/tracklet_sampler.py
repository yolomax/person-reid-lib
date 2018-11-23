from lib.utils.util import ConstType, np_filter
import numpy as np


class TrackletSampler(ConstType):
    def __init__(self, track_info, npr):
        self.npr = npr
        self.track_info = track_info

    def _get_train(self, person_id, num):
        tracklet = []
        person_data = np_filter(self.track_info, [person_id])
        cam_id = np.unique(person_data[:, 1])
        cam_num = cam_id.size
        if num > cam_num:
            for sample_i in range(cam_num):
                cam_i = cam_id[sample_i]
                person_cam_i = np_filter(person_data, [person_id], [cam_i])
                track_i = self.npr.randint(0, person_cam_i.shape[0])
                tracklet.append(person_cam_i[track_i, ...])
            for sample_i in range(num - cam_num):
                cam_i = cam_id[self.npr.randint(0, cam_num)]
                person_cam_i = np_filter(person_data, [person_id], [cam_i])
                track_i = self.npr.randint(0, person_cam_i.shape[0])
                tracklet.append(person_cam_i[track_i, ...])
        else:
            cam_order = self.npr.permutation(cam_num)
            for sample_i in range(num):
                cam_i = cam_id[cam_order[sample_i]]
                person_cam_i = np_filter(person_data, [person_id], [cam_i])
                track_i = self.npr.randint(0, person_cam_i.shape[0])
                tracklet.append(person_cam_i[track_i, ...])
        return tracklet

    def get_train_sample(self, id_list, num):
        tracklet = []
        for person_id in id_list:
            tracklet.extend(self._get_train(person_id, num))
        return tracklet