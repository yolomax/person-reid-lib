from __future__ import absolute_import

from .samplerbase import SamplerBase
import numpy as np
import copy
from lib.utils.util import np_filter


class AllTripletSampler(SamplerBase):
    def init(self):
        self.pids = np.unique(self.data_info[:, 0]).tolist()
        self.cam_num = np.unique(self.data_info[:, 1]).size
        self.info_dict = {}

        self._len = 0
        for pid in self.pids:
            p_info = np_filter(self.data_info, [pid])
            cams = np.unique(p_info[:, 1]).tolist()
            self.info_dict[pid] = {}
            self.info_dict[pid]['avai_cams'] = cams
            self.info_dict[pid]['avai_num'] = p_info.shape[0]
            for cam_i in cams:
                p_cam_info = np_filter(p_info, [pid], [cam_i])
                tmp_p_cam_info = []
                p_cam_track_idx = self.npr.permutation(p_cam_info.shape[0])
                for p_cam_i in p_cam_track_idx:
                    tmp_p_cam_info.append(p_cam_info[p_cam_i])
                self.info_dict[pid][cam_i] = tmp_p_cam_info
            num = p_info.shape[0]
            if num < self.sample_num:
                num = self.sample_num
            self._len += num - num % self.sample_num

    def __iter__(self):
        avai_pids = copy.deepcopy(self.pids)
        final_info = []

        while len(avai_pids) >= self.instance_num:
            batch_info = []
            selected_pids = self.npr.choice(avai_pids, self.instance_num, replace=False)
            for pid in selected_pids:
                if len(self.info_dict[pid]['avai_cams']) > self.sample_num:
                    selected_cams = self.npr.choice(self.info_dict[pid]['avai_cams'], self.sample_num,
                                                    replace=False).tolist()
                    for selected_cam_i in selected_cams:
                        batch_info.append(self.info_dict[pid][selected_cam_i].pop())
                        if len(self.info_dict[pid][selected_cam_i]) == 0:
                            self.info_dict[pid]['avai_cams'].remove(selected_cam_i)
                        self.info_dict[pid]['avai_num'] -= 1
                else:
                    if self.info_dict[pid]['avai_num'] < self.sample_num:
                        tmp_data_info = []
                        while len(self.info_dict[pid]['avai_cams']) > 0:
                            selected_cam_i = self.info_dict[pid]['avai_cams'].pop()
                            tmp_data_info.extend(self.info_dict[pid][selected_cam_i])
                            self.info_dict[pid]['avai_num'] -= len(self.info_dict[pid][selected_cam_i])
                            self.info_dict[pid][selected_cam_i] = []

                        tmp_data_info_new_idx = self.npr.choice(len(tmp_data_info), self.sample_num - len(tmp_data_info), replace=True)
                        for tmp_data_info_i in tmp_data_info_new_idx:
                            batch_info.append(tmp_data_info[tmp_data_info_i])
                        batch_info.extend(tmp_data_info)
                    else:
                        tmp_data_info = []
                        while True:
                            for selected_cam_i in copy.deepcopy(self.info_dict[pid]['avai_cams']):
                                tmp_data_info.append(self.info_dict[pid][selected_cam_i].pop())
                                if len(self.info_dict[pid][selected_cam_i]) == 0:
                                    self.info_dict[pid]['avai_cams'].remove(selected_cam_i)
                                self.info_dict[pid]['avai_num'] -= 1
                                if len(tmp_data_info) == self.sample_num:
                                    break
                            if len(tmp_data_info) == self.sample_num:
                                batch_info.extend(tmp_data_info)
                                break
                    if self.info_dict[pid]['avai_num'] == 0:
                        avai_pids.remove(pid)
            final_info.append(batch_info)

        track_info = []
        for batch_info in final_info:
            track_info.append(self._process_raw_data(batch_info))

        return iter(track_info)




