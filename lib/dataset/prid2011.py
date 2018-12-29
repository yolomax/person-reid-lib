import numpy as np
import glob
import os.path as osp
from scipy.io import loadmat
from urllib.request import urlretrieve
from .datasetbase import DataSetBase
from lib.utils.util import np_filter, unpack_file, check_path


class PRID2011(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('PRID-2011', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'prid_2011.zip'

        self.raw_data_folder = self.store_dir / 'prid_2011'
        self.dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
        self.split_mat_path = root_dir / 'iLIDS-VID/train-test people splits/train_test_splits_prid.mat'
        self.cam_a_path = self.raw_data_folder / 'multi_shot' / 'cam_a'
        self.cam_b_path = self.raw_data_folder / 'multi_shot' / 'cam_b'

        self.split_rate = 0.5

        self.minframes = 27
        self.npr = npr
        self.resize_hw = None
        self.init()

    def check_raw_file(self):
        if not self.zipfiles_dir.exists():
            check_path(self.zipfiles_dir.parent, create=True)
            urlretrieve(self.dataset_url, self.zipfiles_dir)
        if not self.raw_data_folder.exists():
            check_path(self.raw_data_folder, create=True)
            unpack_file(self.zipfiles_dir, self.raw_data_folder, self.logger)

        assert self.split_mat_path.exists()

    def _get_dict(self):
        self.logger.info('Begin Get Video List')
        assert self.cam_a_path.exists() and self.cam_b_path.exists()

        person_cama_dirs = sorted(glob.glob(osp.join(self.cam_a_path, '*')))[:200]
        person_camb_dirs = sorted(glob.glob(osp.join(self.cam_b_path, '*')))[:200]

        person_cama_dirs = [osp.basename(item) for item in person_cama_dirs]
        person_camb_dirs = [osp.basename(item) for item in person_camb_dirs]
        assert set(person_cama_dirs) == set(person_camb_dirs)

        frames_list = []
        video = np.zeros((400, 5), dtype=np.int64)
        video_id = 0
        frames_begin = 0

        for pid, person in enumerate(person_cama_dirs):
            for cam_i, cam_path in enumerate([self.cam_a_path, self.cam_b_path]):
                frames_name = glob.glob('%s/%s/*.png' % (str(cam_path), person))
                num_frames = len(frames_name)
                frames_name.sort(key=lambda x: int(x[-8:-4]))
                video[video_id, 0] = pid
                video[video_id, 1] = cam_i
                video[video_id, 2] = frames_begin
                video[video_id, 3] = frames_begin + num_frames
                video[video_id, 4] = num_frames
                video_id += 1
                frames_list.extend(frames_name)
                frames_begin = frames_begin + num_frames

        splits = self._prepare_split(video.copy())

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
            data_split['info'] = 'PRID-2011 dataset. Split ID {:2d}'.format(split_id)

            train_id = np.unique(data_split['train'][:, 0])
            probe_id = np.unique(data_split['probe'][:, 0])
            gallery_id = np.unique(data_split['gallery'][:, 0])

            assert np.intersect1d(probe_id, gallery_id).size == probe_id.size
            assert probe_id.size == gallery_id.size
            assert data_split['probe'].shape[0] == data_split['gallery'].shape[0]

            data_splits.append(data_split)

        data_dict['split'] = data_splits
        data_dict['track_info'] = video
        data_dict['info'] = 'PRID-2011 Dataset. Min Frames {:3d}. 10 Splits.'.format(self.minframes)

        return data_dict

    def _prepare_split(self, track_info):
        if self.minframes == 27:
            return self._load_from_mat(track_info)
        elif self.minframes == 21:
            return self._create_new(track_info)
        else:
            raise ValueError

    def person_filter(self, all_track_raw, minframes):
        track_info = all_track_raw[all_track_raw[:, 4] >= minframes]
        person_id = np.unique(track_info[:, 0])
        person_id_box = []
        for i_person in range(person_id.size):
            person_data = np_filter(track_info, [person_id[i_person]])
            person_cam = np.unique(person_data[:, 1])
            if person_cam.size >= 2:
                person_id_box.append(person_id[i_person])
        return np.asarray(person_id_box)

    def _create_new(self, track_info):
        self.logger.info("Create new splits for PRID-2011 with min frames length 21")
        num_splits = 10
        train_test_idx = self.person_filter(track_info, 21)
        train_test_idx.sort()
        num_total_ids = train_test_idx.size
        assert num_total_ids == 183
        num_ids_each = int(num_total_ids * self.split_rate)

        splits = []
        for i_split in range(num_splits):
            # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
            person_idx = self.npr.permutation(num_total_ids)
            train_idxs = person_idx[num_ids_each:]
            test_idxs = person_idx[:num_ids_each]
            train_idxs.sort()
            test_idxs.sort()

            split = {'train': train_test_idx[train_idxs], 'test': train_test_idx[test_idxs]}
            splits.append(split)

        self.logger.info("Totally {} splits are created.".format(len(splits)))
        return splits

    def _load_from_mat(self, track_info):
        self.logger.info("Load splits from mat file <--- " + str(self.split_mat_path))
        mat_split_data = loadmat(self.split_mat_path)['ls_set']

        num_splits = mat_split_data.shape[0]
        num_total_ids = mat_split_data.shape[1]
        assert num_splits == 10
        assert num_total_ids == 178

        train_test_idx = self.person_filter(track_info, 27)
        train_test_idx.sort()
        assert train_test_idx.size == 178

        num_ids_each = int(num_total_ids * self.split_rate)

        splits = []
        for i_split in range(num_splits):
            # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
            train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
            test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))

            train_idxs = [int(i) - 1 for i in train_idxs]
            test_idxs = [int(i) - 1 for i in test_idxs]
            train_idxs.sort()
            test_idxs.sort()

            split = {'train': train_test_idx[train_idxs], 'test': train_test_idx[test_idxs]}
            splits.append(split)

        self.logger.info("Totally {} splits are created.".format(len(splits)))
        return splits
