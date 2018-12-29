from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import os.path as osp
import re
from .datasetbase import DataSetBase
from lib.utils.util import unpack_file


__all__ = ['Market1501']


class Market1501(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('Market1501', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'Market-1501-v15.09.15.zip'

        self.raw_data_folder = self.store_dir / 'Market-1501-v15.09.15'
        self.train_dir = self.raw_data_folder / 'bounding_box_train'
        self.query_dir = self.raw_data_folder / 'query'
        self.gallery_dir = self.raw_data_folder / 'bounding_box_test'


        self.resize_hw = None
        self.init()

    def check_raw_file(self):
        assert self.zipfiles_dir.exists()
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)

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
        data_split['info'] = 'Market1501 dataset. Split ID {:2d}. Remove Junk Images'.format(0)

        data_dict['split'] = [data_split]
        data_dict['info'] = 'Market1501 Dataset. Remove Junk Images'

        return data_dict

    def _process_dir(self, dir_path, begin_id):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        img_dir_list = []
        idx = begin_id
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            img_dir_list.append(img_path)
            dataset.append((pid, camid, idx, idx+1, 1))
            idx += 1

        return np.asarray(dataset, dtype=np.int64), img_dir_list
