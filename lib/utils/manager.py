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
                 'root': Path('/home/username/data'),      # Store the extracted files
                 'rawfiles': Path('/home/username/rawfiles'),  # The location of the original compressed file
                 'Model': Path('/home/username/model'),    # Store the officially downloaded torch model parameters
                 'web_env_dir': '/home/username/ignore',
                 'web_host': "http://localhost",
                 'web_port': 31094,
                 'num_workers': 4,
                 'test_batch_size': 16},
            'server': {'name': 'server',
                       'root': Path('/data/data'),  # Store the extracted files
                       'rawfiles': Path('/data/rawfiles'),  # The location of the original compressed file
                       'Model': Path('/data/model'),  # Store the officially downloaded torch model parameters
                       'web_env_dir': '/data/ignore',
                       'web_host': "http://localhost",
                       'web_port': 31094,
                       'num_workers': 16,
                       'test_batch_size': 64}
        }

        self._dataset_box = ['iLIDS-VID', 'PRID-2011', 'LPW', 'MARS', 'VIPeR', 'Market1501', 'CUHK03', 'CUHK01',
                             'DukeMTMCreID', 'GRID', 'DukeMTMC-VideoReID']

        self.init_device()

        self.recorder = Recorder(check_path(self.task_dir, create=False), self.time_tag, self.device)
        self.logger = self.recorder.logger
        self.logger.info('Device: ' + self.device['name'])
        self.logger.info('{0:-^60}'.format('Set Seed ' + str(self.seed)))

    def set_dataset(self, idx):
        if isinstance(idx, int):
            self.dataset_name = self._dataset_box[idx]
        elif isinstance(idx, str):
            assert idx in self._dataset_box
            self.dataset_name = idx
        else:
            raise TypeError
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
        if self._device_dict['pc']['rawfiles'].exists():
            device = self._device_dict['pc']
        else:
            device = self._device_dict['server']

        check_path(device['root'], create=True)
        check_path(device['Model'], create=True)
        check_path(device['web_env_dir'], create=True)

        self.device = device

    def store_performance(self, cmc_box):
        file_path = check_path(self.task_dir / 'output/result', create=True) / str('cmc_' + self.time_tag + '.json')
        DataPacker.dump(cmc_box, file_path, self.logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
