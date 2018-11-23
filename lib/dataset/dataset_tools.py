from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import lmdb
import h5py
from pathlib import Path
from PIL import Image
from lib.utils.util import ParseDatasetName, ConstType, check_path


__all__ = ['DataStoreManager']


class DataStoreManager(ConstType):
    def __init__(self, file_folder, dataset_name, store_type, image_dir_list, track_info, resize=None, logger=None):
        self.file_folder = Path(file_folder)
        self.dataset_name = dataset_name
        self.store_type = store_type
        self.len = len(image_dir_list)
        self.resize_hw = resize
        self.logger = logger
        if self.resize_hw is not None:
            assert isinstance(self.resize_hw, (tuple, list))
            assert len(self.resize_hw) == 2
        check_path(self.file_folder, create=True)
        dataset_dir = list(self.file_folder.glob(self.dataset_name + '_*.' + self.store_type))

        self._store_factory = {'db': {'read': self.read_lmdb, 'write': self.write_lmdb},
                               'h5': {'read': self.read_h5, 'write': self.write_h5}}

        self.store_optical_flow = False

        if len(dataset_dir) == 0:
            with_optical_flow = False
            read_img = self.read_only_img
            if track_info is not None and np.max(track_info[:, -1]) > 1 and self.store_optical_flow:
                try:
                    from .optical_flow_tools import OpticalFlowManager
                    self.of_generator = OpticalFlowManager(image_dir_list, track_info, self.resize_hw)
                    read_img = self.read_img_with_of
                    with_optical_flow = True
                except ImportError as e:
                    logger.error(e)

            self.with_optical_flow = with_optical_flow
            self.read_img = read_img
            self.dataset_dir = self.write(image_dir_list)
        else:
            assert len(dataset_dir) == 1
            self.dataset_dir = dataset_dir[0]
        self.parse()
        self.init()

    def __len__(self):
        return self.len

    def init(self):
        if self.store_type == 'h5':
            with h5py.File(self.dataset_dir, 'r') as dataset:
                self.data = dataset['data'][...]
        elif self.store_type == 'db':
            self.lmdb_env = lmdb.open(str(self.dataset_dir), map_size=int(1099511627776), readonly=True,
                                      max_spare_txns=20, max_readers=256, lock=False)

    def write(self, image_dir_list):
        return self._store_factory[self.store_type]['write'](image_dir_list)

    def read(self, item):
        return self._store_factory[self.store_type]['read'](item)

    def get_store_dir(self, img_dir):
        test_img = self.read_only_img(img_dir)
        img_shape = test_img.shape
        if self.resize_hw is not None:
            img_shape = (self.resize_hw[0], self.resize_hw[1], img_shape[-1])
        if self.with_optical_flow:
            img_shape = list(img_shape)
            img_shape[-1] += 2

        img_dtype = test_img.dtype

        name = self.dataset_name + '_' + ParseDatasetName.to_str(img_shape, img_dtype) + '.' + self.store_type
        return self.file_folder / name

    def parse(self):
        self.img_shape, self.img_dtype = ParseDatasetName.recover(self.dataset_dir.name)

    def write_lmdb(self, image_dir_list):
        lmdb_dir = self.get_store_dir(image_dir_list[0])
        img_num = len(image_dir_list)
        img_count = 0
        with lmdb.open(str(lmdb_dir), map_size=int(1099511627776)) as lmdb_env:
            with lmdb_env.begin(write=True) as lmdb_txn:
                self.logger.info('Store database -->' + str(lmdb_dir))
                for im_dir in image_dir_list:
                    img = self.read_img(im_dir, img_count)
                    key_id = '%08d' % img_count
                    lmdb_txn.put(key_id.encode(), img)
                    img_count += 1
                    if img_count % 10000 == 0 or img_count == img_num:
                        self.logger.info('pass %d, key id : %s' % (img_count, key_id))
        assert img_count == img_num
        self.logger.info('Total %08d images. Creating Finish' % img_count)
        return lmdb_dir

    def read_lmdb(self, item):
        output = []

        if isinstance(item, (list, tuple)):
            idx_list = item
        elif isinstance(item, int):
            idx_list = [item]
        elif isinstance(item, slice):
            tmp_start = item.start or 0
            if tmp_start >= self.len:
                tmp_start = self.len
            elif tmp_start <= - self.len:
                tmp_start = 0
            else:
                tmp_start = tmp_start % self.len

            tmp_stop = item.stop or self.len
            if tmp_stop >= self.len:
                tmp_stop = self.len
            elif tmp_stop <= - self.len:
                tmp_stop = 0
            else:
                tmp_stop = tmp_stop % self.len
            tmp_step = item.step or 1
            assert tmp_step > 0
            idx_list = list(range(tmp_start, tmp_stop, tmp_step))
        else:
            raise TypeError

        with self.lmdb_env.begin() as lmdb_txn:
                for idx in idx_list:
                    key_id = '%08d' % idx
                    temp_img = lmdb_txn.get(key_id.encode())
                    temp_img = np.fromstring(temp_img, dtype=self.img_dtype)
                    temp_img = temp_img.reshape(self.img_shape)
                    output.append(temp_img[...])
        return output

    def write_h5(self, image_dir_list):
        file_dir = self.get_store_dir(image_dir_list[0])
        img_list = []
        img_count = 0
        img_num = len(image_dir_list)
        self.logger.info('Begin write h5 file.')
        for im_dir in image_dir_list:
            img = self.read_img(im_dir, img_count)
            img_list.append(img)
            img_count += 1
            if img_count % 500 == 0 or img_count == img_num:
                self.logger.info('pass %d' % img_count)
        dataset = np.asarray(img_list)
        assert img_count == img_num
        with h5py.File(file_dir, 'w') as f:
            self.logger.info('Store database -->' + str(file_dir))
            grp_data = f.create_dataset('data', dataset.shape, data=dataset)
        self.logger.info('Creating Finish')
        return file_dir

    def read_h5(self, item):
        assert isinstance(item, (list, tuple))
        output = []
        for idx in item:
            output.append(self.data[idx, ...].copy())
        return output

    def close(self):
        if self.store_type == 'db':
            self.lmdb_env.close()

    def read_only_img(self, img_dir, idx=None):
        img = Image.open(img_dir).convert('RGB')
        if self.resize_hw is not None:
            img = img.resize([self.resize_hw[1], self.resize_hw[0]], Image.BILINEAR)
        return np.asarray(img)

    def read_img_with_of(self, img_dir, idx):
        img_and_of = self.of_generator(idx)
        return np.asarray(img_and_of)
