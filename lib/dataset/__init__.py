from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lib.utils.util import np_filter
import numpy as np
import copy
from torch.utils.data import Dataset
from .prid2011 import PRID2011
from .ilidsvid import iLIDSVID
from .mars import MARS
from .lpw import LPW
from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .grid import GRID
from .market1501 import Market1501
from .viper import VIPeR
from .dukemtmcvidreid import DukeMTMCVidReID


class ConstType(object):
    class ConstError(TypeError):
        pass

    def __setattr__(self, key, value):
        if key != '_transform' and key in self.__dict__:
            raise self.ConstError
        else:
            self.__dict__[key] = value


class DataBank(Dataset, ConstType):
    def __init__(self, name, root_dir, rawfiles_dir, split_id, npr, minframes=None, logger=None):
        self.name = name
        self._transform = None
        self.logger = logger

        self._dataset_factory = {'PRID-2011': PRID2011,
                                 'iLIDS-VID': iLIDSVID,
                                 'MARS': MARS,
                                 'LPW': LPW,
                                 'CUHK01': CUHK01,
                                 'CUHK03': CUHK03,
                                 'DukeMTMCreID': DukeMTMCreID,
                                 'GRID': GRID,
                                 'Market1501': Market1501,
                                 'VIPeR': VIPeR,
                                 'DukeMTMC-VideoReID': DukeMTMCVidReID}

        if not self.name in self._dataset_factory:
            self.logger.error('No this dataset. Wrong name.')
            raise KeyError

        if self.name in ['PRID-2011', 'iLIDS-VID', 'MARS', 'LPW', 'DukeMTMC-VideoReID']:
            self.is_image_dataset = False
            self.minframes = minframes
        else:
            self.is_image_dataset = True
            self.minframes = None

        self._dataset = self._dataset_factory[self.name](root_dir, rawfiles_dir, split_id, npr, self.logger)
        self.images_dir_list = copy.deepcopy(self._dataset.data_dict['dir'])
        self.shape = copy.deepcopy(self._dataset.data_dict['shape'])
        self.train_info, self.test_info, self.probe_index, self.gallery_index, self.junk_index = self._preprocess()

        self.train_person_num = np.unique(self.train_info[:, 0]).size
        self.test_person_num = np.unique(self.test_info[:, 0]).size
        self.train_image_num = np.sum(self.train_info[:, 4])
        self.test_image_num = np.sum(self.test_info[:, 4])

        self.train_frames_len_min = np.min(self.train_info[:, 4])
        self.train_frames_len_max = np.max(self.train_info[:, 4])
        self.test_frames_len_min = np.min(self.test_info[:, 4])
        self.test_frames_len_max = np.max(self.test_info[:, 4])

        self.train_cam_num = np.unique(self.train_info[:, 1]).size
        self.test_cam_num = np.unique(self.test_info[:, 1]).size

    def set_transform(self, transform):
        self._transform = transform

    def _check(self, train_info, test_info, probe_info, gallery_info):
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]

        if self.minframes is not None:
            probe_info_new = []
            probe_info_drop = []
            for probe_i in range(probe_info.shape[0]):
                data_info = probe_info[probe_i]
                p_id = data_info[0]
                p_cam_id = data_info[1]
                g_info = np_filter(gallery_info, [p_id])
                g_cam_id = np.unique(g_info[:, 1])
                if np.setdiff1d(g_cam_id, np.asarray([p_cam_id])).size == 0:
                    probe_info_drop.append(data_info)
                else:
                    probe_info_new.append(data_info)

            self.logger.info('After drop videos less than: train {:2d} test {:2d} frames, check cam number'.format(
                self.minframes['train'], self.minframes['test']))
            if len(probe_info_drop) > 0:
                for drop_info in probe_info_drop:
                    self.logger.info('No related gallery track. Drop probe ' + str(drop_info))
                probe_info = np.stack(probe_info_new)
                test_info = self._merge_to_test(probe_info, gallery_info)
            else:
                self.logger.warn('All probe track have related gallery track.')

        assert np.sum(train_info[:, 3] - train_info[:, 2] - train_info[:, 4]) == 0
        assert np.sum(test_info[:, 3] - test_info[:, 2] - test_info[:, 4]) == 0
        assert np.sum(probe_info[:, 3] - probe_info[:, 2] - probe_info[:, 4]) == 0
        assert np.sum(gallery_info[:, 3] - gallery_info[:, 2] - gallery_info[:, 4]) == 0

        test_id = np.unique(test_info[:, 0])
        probe_id = np.unique(probe_info[:, 0])
        gallery_id = np.unique(gallery_info[:, 0])
        
        assert -1 not in set(test_id)   # junk id set to be -1, it should have been removed.
        assert np.setdiff1d(probe_id, gallery_id).size == 0
        assert set(test_id) == set(probe_id).union(set(gallery_id))

        for probe_i in range(probe_info.shape[0]):
            data_info = probe_info[probe_i]
            p_id = data_info[0]
            p_cam_id = data_info[1]
            g_info = np_filter(gallery_info, [p_id])
            g_cam_id = np.unique(g_info[:, 1])
            if not np.setdiff1d(g_cam_id, np.asarray([p_cam_id])).size > 0:
                self.logger.warn('All gallery trackets have the same camera id with probe tracklet for ID： ' + str(p_id))

        assert np.unique(test_info[:, 2]).size == np.unique(np.concatenate((probe_info, gallery_info))[:, 2]).size
        assert np.intersect1d(train_info[:, 2], test_info[:, 2]).size == 0
        assert np.unique(train_info[:, 2]).size == train_info.shape[0]
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]
        assert np.unique(probe_info[:, 2]).size == probe_info.shape[0]
        assert np.unique(gallery_info[:, 2]).size == gallery_info.shape[0]

        return test_info, probe_info

    @staticmethod
    def _get_index(rawset, subset):
        index = []
        for i_probe in range(subset.shape[0]):
            begin = subset[i_probe, 2]
            temp_index = np.where(rawset[:, 2] == begin)[0]
            assert temp_index.size == 1
            temp_index = temp_index[0]
            index.append(temp_index)
        index = np.asarray(index, dtype=np.int64)
        return index

    def _merge_to_test(self, probe_info, gallery_info):
        begin_idx_box = gallery_info[:, 2].tolist()
        temp_info = []
        for probe_i in range(probe_info.shape[0]):
            probe_i_info = probe_info[probe_i]
            if probe_i_info[2] not in begin_idx_box:
                temp_info.append(probe_i_info)
                begin_idx_box.append(probe_i_info[2])

        if len(temp_info) == 0:
            return gallery_info.copy()
        else:
            temp_info = np.asarray(temp_info, np.int64)
            test_info = np.concatenate((temp_info, gallery_info), axis=0)
            return test_info

    def _preprocess(self):
        train_info, probe_info, gallery_info = self._dataset.prepare_data()
        test_info = self._merge_to_test(probe_info, gallery_info)

        self._print_info(train_info, test_info, probe_info, gallery_info)

        if self.minframes is not None:
            train_info = train_info[train_info[:, 4] >= self.minframes['train']]
            test_info = test_info[test_info[:, 4] >= self.minframes['test']]
            probe_info = probe_info[probe_info[:, 4] >= self.minframes['test']]
            gallery_info = gallery_info[gallery_info[:, 4] >= self.minframes['test']]

        test_info, probe_info = self._check(train_info, test_info, probe_info, gallery_info)

        if self.minframes is not None:
            self._print_info(train_info, test_info, probe_info, gallery_info)

        probe_idx = DataBank._get_index(test_info, probe_info)
        gallery_idx = DataBank._get_index(test_info, gallery_info)

        junk_idx = np.where(test_info[:, 0] == -1)[0]
        train_info = self._get_pseudo_label(train_info)
        test_info = self._get_pseudo_label(test_info)
        return train_info, test_info, probe_idx, gallery_idx, junk_idx

    @staticmethod
    def _get_pseudo_label(track_info):
        pseudo_info = track_info.copy()
        real_pid = np.unique(track_info[:, 0])
        real_pid.sort()
        person_num = real_pid.size
        real_cid = np.unique(track_info[:, 1])
        real_cid.sort()
        cam_num = real_cid.size
        for pseudo_id in range(person_num):
            person_real_id = real_pid[pseudo_id]
            pseudo_info[track_info[:, 0] == person_real_id, 0] = pseudo_id
        for pseudo_id in range(cam_num):
            person_real_cid = real_cid[pseudo_id]
            pseudo_info[track_info[:, 1] == person_real_cid, 1] = pseudo_id
        return pseudo_info

    def __getitem__(self, item):
        person_id = item[0]
        cam_id = item[1]
        sample_list = item[2:]
        data = self._dataset.read(sample_list)
        if self._transform is not None:
            data = self._transform(data)
        return data, person_id

    def _print_info(self, train_info, test_info, probe_info, gallery_info):

        GalleryInds = np.unique(gallery_info[:, 0])
        probeInds = np.unique(probe_info[:, 0])

        self.logger.info('           Train     Test    Probe   Gallery')
        self.logger.info('#ID       {:5d}    {:5d}    {:5d}    {:5d}'.format(np.unique(train_info[:, 0]).size,
                                                                             np.unique(test_info[:, 0]).size,
                                                                             np.unique(probe_info[:, 0]).size,
                                                                             np.unique(gallery_info[:, 0]).size))
        self.logger.info('#Track {:8d} {:8d} {:8d} {:8d}'.format(train_info.shape[0],
                                                                 test_info.shape[0],
                                                                 probe_info.shape[0],
                                                                 gallery_info.shape[0]))
        self.logger.info('#Image {:8d} {:8d} {:8d} {:8d}'.format(np.sum(train_info[:, 4]),
                                                                 np.sum(test_info[:, 4]),
                                                                 np.sum(probe_info[:, 4]),
                                                                 np.sum(gallery_info[:, 4])))
        self.logger.info(
            '#Cam        {:2d}       {:2d}       {:2d}       {:2d}'.format(np.unique(train_info[:, 1]).size,
                                                                           np.unique(test_info[:, 1]).size,
                                                                           np.unique(probe_info[:, 1]).size,
                                                                           np.unique(gallery_info[:, 1]).size))
        self.logger.info('MaxLen {:8d} {:8d} {:8d} {:8d}'.format(np.max(train_info[:, 4]),
                                                                 np.max(test_info[:, 4]),
                                                                 np.max(probe_info[:, 4]),
                                                                 np.max(gallery_info[:, 4])))
        self.logger.info('MinLen {:8d} {:8d} {:8d} {:8d}'.format(np.min(train_info[:, 4]),
                                                                 np.min(test_info[:, 4]),
                                                                 np.min(probe_info[:, 4]),
                                                                 np.min(gallery_info[:, 4])))

        self.logger.info('Gallery ID diff Probe ID: %s' % np.setdiff1d(GalleryInds, probeInds))
        self.logger.info('{0:-^60}'.format(''))

    def __len__(self):
        self.logger.info('-------The length of dataset is no meaning!---------')
        return self.train_info.shape[0] + self.test_info.shape[0]
