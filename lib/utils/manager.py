from __future__ import absolute_import
from __future__ import print_function


from pathlib import Path
from lib.utils.util import empty_folder, check_path, DataPacker
from lib.utils.meter import time_tag
from lib.recorder import Recorder


class Manager(object):
    def __init__(self, task_dir, seed=1, mode='Train'):
        self.seed = seed
        self.task_dir = task_dir
        self.time_tag = time_tag()
        self.mode = mode

        self._device_dict = {
            'pc':
                {'name': 'pc',
                 'data': Path('/data/data/data/'),
                 'rawfile': Path('/data//data/rawfile'),
                 'Model': Path('/home/username/opt/model'),
                 'web_env_dir': '/home/username/ignore',
                 'web_host': "http://localhost",
                 'web_port': 31094,
                 'num_workers': 4,
                 'test_batch_size': 16},
            'server': {'name': 'server',
                       'data': Path('/data1/username/data'),
                       'rawfile': Path('/data1/username/rawfile'),
                       'Model': Path('/data1/username/model'),
                       'web_env_dir': '/home/username/ignore',
                       'web_host': "http://localhost",
                       'web_port': 31094,
                       'num_workers': 16,
                       'test_batch_size': 64}
        }

        self._dataset_box = ['iLIDS-VID', 'PRID-2011', 'LPW', 'MARS', 'VIPeR', 'Market1501', 'CUHK03', 'CUHK01',
                             'DukeMTMCreID', 'GRID']

        self.init_device()

        self.recorder = Recorder(check_path(self.task_dir, create=False), self.time_tag, self.device)
        self.logger = self.recorder.logger
        self.logger.info('{0:-^60}'.format('Set Seed ' + str(self.seed)))

    def set_dataset(self, idx):
        self.dataset_name = self._dataset_box[idx]
        self.logger.info('{0:-^60}'.format('Set Dataset ' + self.dataset_name))

    def check_epoch(self, epoch_id):
        with open(check_path(self.task_dir / 'test_epoch_id.txt', create=False)) as f:
            info_list_tmp = f.readlines()
            info_list = [int(info_i) for info_i in info_list_tmp if info_i != '\n']

            assert len(info_list) > 0
            idx_mode = info_list[0]
            if idx_mode == 0:
                assert len(info_list) >= 4
                idx_list = list(range(info_list[1], info_list[2], info_list[3]))
                idx_list += info_list[4:]
                epoch_size = info_list[2]
                idx_list += [epoch_size]

            elif idx_mode == 1:
                idx_list = info_list[1:]
                epoch_size = max(info_list[1:])
            else:
                raise ValueError

            train_flag = True
            test_flag = False

            if epoch_id >= epoch_size:
                train_flag = False
            if epoch_id in idx_list:
                test_flag = True
            return train_flag, test_flag

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value == 'Train':
            self._mode = value
            empty_folder(check_path(self.task_dir / 'output/log', create=True))
            empty_folder(check_path(self.task_dir / 'output/model', create=True))
            empty_folder(check_path(self.task_dir / 'output/result', create=True))
        elif value == 'Test':
            self._mode = value
        else:
            raise KeyError

    @property
    def split_id(self):
        return self._split_id

    @split_id.setter
    def split_id(self, value):
        assert isinstance(value, int)
        self._split_id = value
        if self.dataset_name == 'CUHK03' and self.split_id > 1:
            self.cuhk03_classic = True
        else:
            self.cuhk03_classic = False
        self.logger.info('Set Split ID {:2d}'.format(value))

    def init_device(self):
        if self._device_dict['pc']['rawfile'].exists():
            device = self._device_dict['pc']
        elif self._device_dict['server']['rawfile'].exists():
            device = self._device_dict['server']
        else:
            raise KeyError

        check_path(device['data'], create=True)
        check_path(device['Model'], create=True)
        check_path(device['web_env_dir'], create=True)

        device['data_path'] = {'iLIDS-VID': {'raw_file': device['rawfile'] / 'iLIDS-VID.tar',
                                             'folder_path': device['data'] / 'iLIDS-VID'},
                               'LPW': {'raw_file': device['rawfile'] / 'pep_256x128.zip',
                                       'folder_path': device['data'] / 'LPW'},
                               'PRID-2011': {'raw_file': device['rawfile'] / 'prid_2011.zip',
                                             'split_file': device[
                                                               'data'] / 'iLIDS-VID/train-test people splits/train_test_splits_prid.mat',
                                             'folder_path': device['data'] / 'PRID-2011'},
                               'MARS': {'raw_file': device['rawfile'] / 'bbox_train.zip',
                                        'folder_path': device['data'] / 'MARS'},
                               'Market1501': {'raw_file': device['rawfile'] / 'Market-1501-v15.09.15.zip',
                                              'folder_path': device['data'] / 'Market1501'},
                               'CUHK03': {'raw_file': device['rawfile'] / 'cuhk03_release.zip',
                                          'folder_path': device['data'] / 'CUHK03'},
                               'DukeMTMCreID': {'raw_file': device['rawfile'] / 'DukeMTMC-reID.zip',
                                                'folder_path': device['data'] / 'DukeMTMCreID'},
                               'CUHK01': {'raw_file': device['rawfile'] / 'CUHK01.zip',
                                          'folder_path': device['data'] / 'CUHK01'},
                               'VIPeR': {'raw_file': device['rawfile'] / 'VIPeR.v1.0.zip',
                                         'folder_path': device['data'] / 'VIPeR'},
                               'GRID': {'raw_file': device['rawfile'] / 'underground_reid.zip',
                                        'folder_path': device['data'] / 'GRID'}
                               }


        self.device = device

    def store_performance(self, cmc_box):
        file_path = check_path(self.task_dir / 'output/result', create=True) / str('cmc_' + self.time_tag + '.json')
        DataPacker.dump(cmc_box, file_path, self.logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
