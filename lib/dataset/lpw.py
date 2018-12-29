from lib.dataset.datasetbase import DataSetBase
from lib.utils.util import unpack_file
import numpy as np


__all__ = ['LPW']


class LPW(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('LPW', split_id, 'db', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'pep_256x128.zip'

        self.raw_data_folder = self.store_dir / 'pep_256x128'

        self.resize_hw = None
        self.init()

    def check_raw_file(self):
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir)

    def _get_dict(self):
        frame_list = []
        frame_begin = 0

        raw_data_all = []
        id_num = 0
        cam_num = 0
        for scene_id in range(1, 4):
            scene_data_raw = []
            scene_path = self.raw_data_folder / ('scen' + str(scene_id))
            cam_box = sorted([x for x in scene_path.iterdir() if x.is_dir()])
            for cam_dir in cam_box:
                person_box = sorted([int(x.parts[-1]) for x in cam_dir.iterdir() if x.is_dir()])
                for person_id in person_box:
                    frames_box = cam_dir.glob(str(person_id) + '/*.jpg')
                    frames_id = sorted([int(x.name[:-4]) for x in frames_box])
                    frames_box = [str(cam_dir / str(person_id) / (str(x) + '.jpg')) for x in frames_id]
                    frames_num = len(frames_box)
                    frame_list.extend(frames_box)
                    scene_data_raw.append([person_id + id_num, cam_num, frame_begin, frame_begin + frames_num, frames_num])
                    frame_begin += frames_num
                cam_num += 1
            scene_data_raw = np.asarray(scene_data_raw)
            id_num += scene_data_raw[:, 0].max()
            raw_data_all.append(scene_data_raw)
        track_data = np.concatenate(raw_data_all, axis=0)

        view_data = []
        for cam_id in range(11):
            view_data.append(track_data[track_data[:, 1] == cam_id])

        probe_data = view_data[1]
        gallery_data = np.concatenate((view_data[0], view_data[2]), axis=0)
        train_data = np.concatenate(view_data[3:], axis=0)

        data_dict = {}
        data_dict['dir'] = frame_list
        lpw_dict = {}
        lpw_dict['train'] = train_data
        lpw_dict['probe'] = probe_data
        lpw_dict['gallery'] = gallery_data
        lpw_dict['info'] = 'LPW dataset. Split ID {:2d}'.format(0)
        data_dict['split'] = [lpw_dict]
        data_dict['track_info'] = track_data
        data_dict['info'] = 'LPW Dataset'
        return data_dict
