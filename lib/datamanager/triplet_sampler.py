from __future__ import absolute_import

from .samplerbase import SamplerBase
from lib.datamanager.tracklet_sampler import TrackletSampler
import numpy as np
import copy
from collections import defaultdict


class EqualTripletSampler(SamplerBase):
    def init(self):
        self.tracklet_sampler = TrackletSampler(self.train_info, self.npr)
        self.order = self._get_order(self.person_num, self.instance_num)
        self.order_len = self.order.size
        self._len = self.order.size // self.instance_num
        self.idx = 0

    def _get_order(self, person_num, instance_num):
        id_order = np.arange(person_num)
        self.npr.shuffle(id_order)
        if person_num % instance_num == 0:
            return id_order

        last_begin = person_num - person_num % instance_num
        absent_num = instance_num - person_num % instance_num

        last_id = id_order[last_begin:]
        new_order = np.arange(person_num)
        self.npr.shuffle(new_order)
        new_order = np.setdiff1d(new_order, last_id)
        final_order = np.concatenate((id_order, new_order[:absent_num]))

        assert final_order.size % instance_num == 0
        return final_order

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.order_len:
            self.is_end = True
            raise StopIteration

        tracklet = []
        id_list = self.order[range(self.idx, self.idx + self.instance_num)]
        self.idx += self.instance_num

        tracklet.extend(self.tracklet_sampler.get_train_sample(id_list, self.sample_num))
        return self._process_raw_data(tracklet)


class AllTripletSampler(SamplerBase):
    def init(self):
        self.info_dict = defaultdict(list)

        for idx, track in enumerate(self.train_info):
            self.info_dict[track[0]].append(idx)
        self.pids = list(self.info_dict.keys())

        self._len = 0
        for pid in self.pids:
            idxs = self.info_dict[pid]
            num = len(idxs)
            if num < self.sample_num:
                num = self.sample_num
            self._len += num - num % self.sample_num

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.info_dict[pid])
            if len(idxs) < self.sample_num:
                new_idxs = self.npr.choice(idxs, self.sample_num - len(idxs), replace=True)
                idxs.extend(new_idxs)

            self.npr.shuffle(idxs)

            cam_dict = defaultdict(list)
            for idx in idxs:
                track_info = self.train_info[idx]
                cam_dict[track_info[1]].append(idx)
            avai_cams = list(cam_dict.keys())
            new_idxs = []
            while len(avai_cams) > 0:
                for cam_i in avai_cams:
                    new_idxs.append(cam_dict[cam_i].pop())
                    if len(cam_dict[cam_i]) == 0:
                        avai_cams.remove(cam_i)

            batch_idxs = []
            for idx in new_idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.sample_num:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.instance_num:
            batch_idxs = []
            selected_pids = self.npr.choice(avai_pids, self.instance_num, replace=False)
            for pid in selected_pids:
                batch_idxs_tmp = batch_idxs_dict[pid].pop()
                batch_idxs.extend(batch_idxs_tmp)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            final_idxs.append(batch_idxs)

        track_info = []
        for batch_i in final_idxs:
            batch_info = []
            for idx in batch_i:
                batch_info.append(self.train_info[idx])
            track_info.append(self._process_raw_data(batch_info))

        return iter(track_info)


class OldTripletSampler(SamplerBase):
    def init(self):
        self.raw_info_dict = defaultdict(list)

        for idx, track in enumerate(self.train_info):
            self.raw_info_dict[track[0]].append(idx)

        self.info_dict = copy.deepcopy(self.raw_info_dict)
        self.pids = list(self.info_dict.keys())

        self._len = 0
        for pid in self.pids:
            idxs = self.info_dict[pid]
            num = len(idxs)
            if num < 2:
                num = 2
            self._len += num - num % 2

    def get_negative(self, selected_pids):
        remaining_ids = np.setdiff1d(self.pids, selected_pids)
        negative_pids = self.npr.choice(remaining_ids, self.sample_num, replace=False)
        idx_box = []
        for pid in negative_pids:
            idx_box.extend(self.npr.choice(self.raw_info_dict[pid], 1, replace=False).tolist())
        return idx_box

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.info_dict[pid])
            if len(idxs) < 2:
                new_idxs = self.npr.choice(idxs, 2 - len(idxs), replace=True)
                idxs.extend(new_idxs)

            self.npr.shuffle(idxs)

            cam_dict = defaultdict(list)
            for idx in idxs:
                track_info = self.train_info[idx]
                cam_dict[track_info[1]].append(idx)
            avai_cams = list(cam_dict.keys())
            new_idxs = []
            while len(avai_cams) > 0:
                for cam_i in avai_cams:
                    new_idxs.append(cam_dict[cam_i].pop())
                    if len(cam_dict[cam_i]) == 0:
                        avai_cams.remove(cam_i)

            batch_idxs = []
            for idx in new_idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == 2:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.instance_num:
            batch_idxs = []
            selected_pids = self.npr.choice(avai_pids, self.instance_num, replace=False)
            for pid in selected_pids:
                batch_idxs_tmp = batch_idxs_dict[pid].pop()
                negative_idx = self.get_negative(selected_pids)
                batch_idxs.extend(batch_idxs_tmp)
                batch_idxs.extend(negative_idx)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            final_idxs.append(batch_idxs)

        track_info = []
        for batch_i in final_idxs:
            batch_info = []
            for idx in batch_i:
                batch_info.append(self.train_info[idx])
            track_info.append(self._process_raw_data(batch_info))

        return iter(track_info)



