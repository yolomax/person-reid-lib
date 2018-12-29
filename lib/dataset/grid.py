from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import os.path as osp
from scipy.io import loadmat
from urllib.request import urlretrieve
from .datasetbase import DataSetBase
from lib.utils.util import unpack_file, np_filter, check_path


__all__ = ['GRID']


class GRID(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('GRID', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'underground_reid.zip'

        self.raw_data_folder = self.store_dir / 'underground_reid'
        self.dataset_url = 'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip'
        self.probe_path = self.raw_data_folder / 'probe'
        self.gallery_path = self.raw_data_folder / 'gallery'
        self.split_mat_path = self.raw_data_folder / 'features_and_partitions.mat'

        self.resize_hw = (256, 128)
        self.npr = npr
        self.init()

    def check_raw_file(self):
        if not self.zipfiles_dir.exists():
            check_path(self.zipfiles_dir.parent, create=True)
            urlretrieve(self.dataset_url, self.zipfiles_dir)
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.zipfiles_dir, self.logger)

    def _get_dict(self):
        probe_img_paths = sorted(glob.glob(osp.join(self.probe_path, '*.jpeg')))
        gallery_img_paths = sorted(glob.glob(osp.join(self.gallery_path, '*.jpeg')))

        img_list = []
        images_info = []
        idx = 0
        for img_paths in [probe_img_paths, gallery_img_paths]:
            images_info_tmp = []
            for img_path in img_paths:
                img_list.append(img_path)
                img_name = osp.basename(img_path)
                img_idx = int(img_name.split('_')[0])
                camid = int(img_name.split('_')[1])
                images_info_tmp.append([img_idx, camid, idx, idx + 1, 1])
                idx += 1
            images_info.append(np.asarray(images_info_tmp, dtype=np.int64))

        splits = self._prepare_split(images_info)

        data_dict = {}
        data_dict['dir'] = img_list
        data_dict['split'] = splits
        data_dict['info'] = 'GRID Dataset. 10 Splits.'

        return data_dict

    def _prepare_split(self, images_info):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        self.logger.info("Begin Load 10 random splits")
        split_mat = loadmat(self.split_mat_path)
        trainIdxAll = split_mat['trainIdxAll'][0]  # length = 10

        probe_info_all = images_info[0]
        gallery_info_all = images_info[1]
        probe_id_all = np.unique(probe_info_all[:, 0])
        gallery_id_all = np.unique(gallery_info_all[:, 0])

        splits = []
        for split_idx in range(10):
            train_idxs = trainIdxAll[split_idx][0][0][2][0]
            assert train_idxs.size == 125
            probe_id = np.setdiff1d(probe_id_all, train_idxs)
            gallery_id = np.setdiff1d(gallery_id_all, train_idxs)
            train_info = np.concatenate(
                (np_filter(probe_info_all, train_idxs), np_filter(gallery_info_all, train_idxs)),
                axis=0)
            probe_info = np_filter(probe_info_all, probe_id)
            gallery_info = np_filter(gallery_info_all, gallery_id)

            assert np.intersect1d(probe_id, gallery_id).size == probe_id.size
            assert probe_id.size == 125
            assert gallery_id.size == 126

            split = {}
            split['train'] = train_info
            split['probe'] = probe_info
            split['gallery'] = gallery_info
            split['info'] = 'GRID dataset. Split ID {:2d}'.format(split_idx)
            splits.append(split)
        self.logger.info("Load 10 random splits")
        return splits
