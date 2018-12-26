from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lib.utils.util import check_path, ConstType, DataPacker
from lib.dataset.utils import DataStoreManager
import numpy as np


class DataSetBase(ConstType):
    def __init__(self, name, split_id, store_type, root_dir, logger):
        self.name = name
        self.split_id = split_id
        self.store_type = store_type
        self.store_dir = root_dir / str(name)
        self.logger = logger
        self.dict_dir = check_path(self.store_dir, create=True) / (str(name) + '_dict.json')

    def _get_dict(self):
        raise NotImplementedError

    def check_raw_file(self):
        raise NotImplementedError

    def _store_dict(self, data_dict):
        DataPacker.dump(data_dict, self.dict_dir, self.logger)

    def _read_dict(self, split_id=0):
        if not self.dict_dir.exists():
            self.check_raw_file()
            data_dict = self._get_dict()
            self._store_dict(data_dict)

        tmp_data_dict = DataPacker.load(self.dict_dir, self.logger)
        self.logger.info(tmp_data_dict['info'])
        images_dir_list = tmp_data_dict['dir']
        if 'track_info' in tmp_data_dict:
            all_track_info = tmp_data_dict['track_info']
        else:
            all_track_info = None

        assert split_id < len(tmp_data_dict['split'])
        tmp_data_dict = tmp_data_dict['split'][split_id]
        self.logger.info('The choosed split info: ' + tmp_data_dict['info'])

        data_dict = {}
        data_dict['train'] = np.asarray(tmp_data_dict['train'], np.int64)
        data_dict['probe'] = np.asarray(tmp_data_dict['probe'], np.int64)
        data_dict['gallery'] = np.asarray(tmp_data_dict['gallery'], np.int64)
        data_dict['dir'] = images_dir_list
        self.storemanager = DataStoreManager(self.store_dir, self.name, self.store_type, data_dict['dir'], all_track_info, self.resize_hw, self.logger)
        data_dict['shape'] = self.storemanager.img_shape
        data_dict['dtype'] = self.storemanager.img_dtype
        self.data_dict = data_dict

    def prepare_data(self):
        train_info = self.data_dict['train']
        probe_info = self.data_dict['probe']
        gallery_info = self.data_dict['gallery']

        self.logger.info('{0:-^60}'.format(self.name + ' Raw dataset with image shape ' + str(self.data_dict['shape'])))

        return train_info, probe_info, gallery_info

    def init(self):
        self._read_dict(self.split_id)
        self.read = self.storemanager.read