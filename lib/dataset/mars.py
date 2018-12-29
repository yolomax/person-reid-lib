import scipy.io as scio
import numpy as np
from lib.dataset.datasetbase import DataSetBase
from lib.utils.util import unpack_file, check_path


__all__ = ['MARS']


class MARS(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('MARS', split_id, 'db', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'bbox_train.zip'

        self.raw_data_folder = self.store_dir / 'MARS'

        self.resize_hw = None
        self.init()

    def check_raw_file(self):
        check_path(self.raw_data_folder, create=True)
        raw_file_folder = self.zipfiles_dir.parent
        raw_data_list = ['bbox_train', 'bbox_test', 'MARS-evaluation-master']

        for raw_data in raw_data_list:
            if not (self.raw_data_folder / raw_data).exists():
                unpack_file(raw_file_folder / str(raw_data + '.zip'), self.raw_data_folder, self.logger)

    def _get_dict(self):
        self.logger.info('Begin Get Video List')
        
        train_name_path = self.raw_data_folder / 'MARS-evaluation-master/info/train_name.txt'
        test_name_path = self.raw_data_folder / 'MARS-evaluation-master/info/test_name.txt'
        track_train_info_path = self.raw_data_folder / 'MARS-evaluation-master/info/tracks_train_info.mat'
        track_test_info_path = self.raw_data_folder / 'MARS-evaluation-master/info/tracks_test_info.mat'
        quary_idx_path = self.raw_data_folder / 'MARS-evaluation-master/info/query_IDX.mat'

        assert train_name_path.exists() and test_name_path.exists()
        assert track_train_info_path.exists() and track_test_info_path.exists()
        assert quary_idx_path.exists()

        train_names = self._get_names(train_name_path)
        test_names = self._get_names(test_name_path)
        track_train = scio.loadmat(track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = scio.loadmat(track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_idx = scio.loadmat(quary_idx_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        track_train[:, 0] -= 1
        track_train[:, 3] -= 1
        track_test[:, 0] -= 1
        track_test[:, 3] -= 1
        query_idx -= 1

        train_track, train_img_dir_list = self._process_data(train_names, track_train, 'bbox_train', 0)
        test_track, test_img_dir_list, query_track = self._process_data(test_names, track_test, 'bbox_test', len(train_img_dir_list), query_idx)
        frames_list = train_img_dir_list + test_img_dir_list

        assert train_track[:, -1].sum() + test_track[:, -1].sum() == len(frames_list)
        assert train_track[-1, 3] == test_track[0, 2] and test_track[-1, 3] == len(frames_list)
        assert train_track[0, 2] == 0
        assert query_idx.shape[0] == query_track.shape[0]

        data_dict = {}
        data_dict['dir'] = frames_list
        mars_dict = {}
        mars_dict['train'] = train_track
        mars_dict['probe'] = query_track
        mars_dict['gallery'] = test_track
        mars_dict['info'] = 'MARS dataset. Split ID {:2d}'.format(0)
        data_dict['split'] = [mars_dict]
        data_dict['track_info'] = np.concatenate((train_track, test_track), axis=0)
        data_dict['info'] = 'MARS Dataset. Remove junk tracklets.'
        return data_dict

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir, begin_id, query_idx=None):
        assert meta_data.shape[1] == 4
        assert home_dir in ['bbox_train', 'bbox_test']
        img_dir_list = []
        track_info = []
        query_info = []

        num_tracklets = meta_data.shape[0]
        if query_idx is not None:
            query_idx = query_idx.tolist()

        idx = begin_id

        for tracklet_idx in range(num_tracklets):

            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data

            if pid == -1:
                continue
            assert 0 <= camid <= 5

            track_info_tmp = [pid, camid, idx, idx+end_index - start_index, end_index - start_index]
            img_names = names[start_index: end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [str(self.raw_data_folder / str(home_dir + '/' + str(img_name[:4]) + '/' + str(img_name))) for img_name in img_names]
            img_dir_list.extend(img_paths)
            track_info.append(track_info_tmp)
            idx += end_index - start_index

            if query_idx is not None and tracklet_idx in query_idx:
                query_info.append(track_info_tmp)

        track_info = np.asarray(track_info, dtype=np.int64)
        if query_idx is not None:
            query_info = np.asarray(query_info, dtype=np.int64)
            return track_info, img_dir_list, query_info
        else:
            return track_info, img_dir_list
