from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import os.path as osp
from .datasetbase import DataSetBase
from lib.utils.util import unpack_file, np_filter


__all__ = ['CUHK01']


class CUHK01(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('CUHK01', split_id, 'h5', root_dir, logger)
        self.raw_data_folder = self.store_dir / 'campus'
        self.zipfiles_dir = rawfiles_dir / 'CUHK01.zip'

        self.resize_hw = None
        self.npr = npr
        self.init()

    def check_raw_file(self):
        assert self.zipfiles_dir.exists()
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)

    def _get_dict(self):
        img_list = sorted(glob.glob(osp.join(self.raw_data_folder, '*.png')))
        images_info = []

        for idx, img_path in enumerate(img_list):
            img_name = osp.basename(img_path)
            pid = int(img_name[:4]) - 1
            camid = (int(img_name[4:7]) - 1) // 2
            images_info.append([pid, camid, idx, idx+1, 1])

        images_info = np.asarray(images_info, dtype=np.int64)
        splits = self._create_split(images_info)

        data_dict = {}
        data_dict['dir'] = img_list
        data_splits = []

        for split_id, split_i in enumerate(splits):
            data_split = {}
            train_idx = split_i['train']
            test_idx = split_i['test']
            data_split['train'] = np_filter(images_info, train_idx)
            data_split['probe'] = np_filter(images_info, test_idx)
            data_split['gallery'] = np_filter(images_info, test_idx)
            data_split['info'] = 'CUHK01 dataset. Split ID {:2d}'.format(split_id)

            probe_id = np.unique(data_split['probe'][:, 0])
            gallery_id = np.unique(data_split['gallery'][:, 0])

            assert np.intersect1d(probe_id, gallery_id).size == probe_id.size
            assert probe_id.size == gallery_id.size
            assert data_split['probe'].shape == data_split['gallery'].shape

            data_splits.append(data_split)

        data_dict['split'] = data_splits
        data_dict['info'] = 'CUHK01 Dataset. 10 Splits.'

        return data_dict

    def _create_split(self, images_info):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        self.logger.info("Creating 10 random splits")

        train_test_idx = np.unique(images_info[:, 0])

        num_total_ids = train_test_idx.size
        num_train_pids = num_total_ids // 2

        splits = []
        for split_id in range(10):
            person_idx = self.npr.permutation(num_total_ids)
            train_idxs = person_idx[:num_train_pids]
            test_idxs = person_idx[num_train_pids:]
            train_idxs.sort()
            test_idxs.sort()

            split = {'train': train_test_idx[train_idxs], 'test': train_test_idx[test_idxs]}
            splits.append(split)

        self.logger.info("Totally {} splits are created.".format(len(splits)))
        return splits
