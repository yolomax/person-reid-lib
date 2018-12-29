from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import os.path as osp
from scipy.io import loadmat
from urllib.request import urlretrieve
from .datasetbase import DataSetBase
from lib.utils.util import np_filter, unpack_file, check_path


__all__ = ['iLIDSVID']


class iLIDSVID(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('iLIDS-VID', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'iLIDS-VID.tar'

        self.raw_data_folder = self.store_dir / 'i-LIDS-VID'
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.split_mat_path = self.store_dir / 'train-test people splits' / 'train_test_splits_ilidsvid.mat'
        self.cam_1_path = self.raw_data_folder / 'sequences/cam1'
        self.cam_2_path = self.raw_data_folder / 'sequences/cam2'
        self.split_rate = 0.5

        self.resize_hw = None
        self.init()

    def check_raw_file(self):
        if not self.zipfiles_dir.exists():
            check_path(self.zipfiles_dir.parent, create=True)
            urlretrieve(self.dataset_url, self.zipfiles_dir)
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)

        assert self.split_mat_path.exists()

    def _get_dict(self):
        assert self.cam_1_path.exists() and self.cam_2_path.exists()

        self.logger.info('Begin Get Video List')
        person_cam1_dirs = sorted(glob.glob(osp.join(self.cam_1_path, '*')))
        person_cam2_dirs = sorted(glob.glob(osp.join(self.cam_2_path, '*')))

        person_cam1_dirs = [osp.basename(item) for item in person_cam1_dirs]
        person_cam2_dirs = [osp.basename(item) for item in person_cam2_dirs]
        assert set(person_cam1_dirs) == set(person_cam2_dirs)

        frames_list = []
        video = np.zeros((600, 5), dtype=np.int64)
        video_id = 0
        frames_begin = 0

        for pid, person in enumerate(person_cam1_dirs):
            for cam_i, cam_path in enumerate([self.cam_1_path, self.cam_2_path]):
                frames_name = glob.glob('%s/%s/*.png' % (str(cam_path), person))
                frames_name.sort(key=lambda x: int(x[-9:-4]))
                num_frames = len(frames_name)
                video[video_id, 0] = pid
                video[video_id, 1] = cam_i
                video[video_id, 2] = frames_begin
                video[video_id, 3] = frames_begin + num_frames
                video[video_id, 4] = num_frames
                video_id += 1
                frames_list.extend(frames_name)
                frames_begin = frames_begin + num_frames

        splits = self._prepare_split()

        data_dict = {}
        data_dict['dir'] = frames_list
        data_splits = []

        for split_id, split_i in enumerate(splits):
            data_split = {}
            train_idx = split_i['train']
            test_idx = split_i['test']
            data_split['train'] = np_filter(video, train_idx)
            data_split['probe'] = np_filter(video, test_idx, [0])
            data_split['gallery'] = np_filter(video, test_idx, [1])
            data_split['info'] = 'iLIDS-VID dataset. Split ID {:2d}'.format(split_id)

            train_id = np.unique(data_split['train'][:, 0])
            probe_id = np.unique(data_split['probe'][:, 0])
            gallery_id = np.unique(data_split['gallery'][:, 0])

            assert np.intersect1d(probe_id, gallery_id).size == 150
            assert train_id.size == 150
            assert data_split['train'].shape[0] == 300
            assert data_split['probe'].shape[0] == 150
            assert data_split['gallery'].shape[0] == 150

            data_splits.append(data_split)

        data_dict['split'] = data_splits
        data_dict['track_info'] = video
        data_dict['info'] = 'iLIDS-VID Dataset. 10 Splits.'

        return data_dict

    def _prepare_split(self):
        self.logger.info("Load splits from mat file <--- " + str(self.split_mat_path))
        mat_split_data = loadmat(self.split_mat_path)['ls_set']

        num_splits = mat_split_data.shape[0]
        num_total_ids = mat_split_data.shape[1]
        assert num_splits == 10
        assert num_total_ids == 300
        num_ids_each = int(num_total_ids * self.split_rate)

        splits = []
        for i_split in range(num_splits):
            # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
            train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
            test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))

            train_idxs = [int(i) - 1 for i in train_idxs]
            test_idxs = [int(i) - 1 for i in test_idxs]

            split = {'train': train_idxs, 'test': test_idxs}
            splits.append(split)

        self.logger.info("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
        return splits
