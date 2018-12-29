from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import re
import os.path as osp
from .datasetbase import DataSetBase
from lib.utils.util import unpack_file, remove_folder


__all__ = ['DukeMTMCreID']


class DukeMTMCreID(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('DukeMTMCreID', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'DukeMTMC-reID.zip'

        self.raw_data_folder = self.store_dir / 'DukeMTMC-reID'
        self.train_dir = self.raw_data_folder / 'bounding_box_train'
        self.query_dir = self.raw_data_folder / 'query'
        self.gallery_dir = self.raw_data_folder / 'bounding_box_test'

        self.resize_hw = (256, 128)
        self.init()

    def check_raw_file(self):
        assert self.zipfiles_dir.exists()
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)
            if (self.store_dir / '__MACOSX').exists():
                remove_folder(self.store_dir / '__MACOSX')

        assert self.train_dir.exists()
        assert self.query_dir.exists()
        assert self.gallery_dir.exists()

    def _get_dict(self):
        img_list = []
        train_images, train_img_dir = self._process_dir(self.train_dir, 0)
        img_list = img_list + train_img_dir
        probe_images, probe_img_dir = self._process_dir(self.query_dir, len(img_list))
        img_list = img_list + probe_img_dir
        gallery_images, gallery_img_dir = self._process_dir(self.gallery_dir, len(img_list))
        img_list = img_list + gallery_img_dir

        train_id = np.unique(train_images[:, 0])
        probe_id = np.unique(probe_images[:, 0])
        gallery_id = np.unique(gallery_images[:, 0])
        assert np.intersect1d(train_id, probe_id).size == 0
        assert np.intersect1d(probe_id, gallery_id).size == probe_id.size
        assert gallery_images[-1, 3] == len(img_list)

        data_dict = {}
        data_dict['dir'] = img_list

        data_split = {}
        data_split['train'] = train_images
        data_split['probe'] = probe_images
        data_split['gallery'] = gallery_images
        data_split['info'] = 'DukeMTMC-reID dataset. Split ID {:2d}.'.format(0)

        data_dict['split'] = [data_split]
        data_dict['info'] = 'DukeMTMCreID Dataset. One Split'

        return data_dict

    def _process_dir(self, dir_path, begin_id):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        img_dir_list = []
        idx = begin_id

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            img_dir_list.append(img_path)
            dataset.append((pid, camid, idx, idx + 1, 1))
            idx += 1

        return np.asarray(dataset, dtype=np.int64), img_dir_list
