import numpy as np
import glob
import os.path as osp
from urllib.request import urlretrieve
from lib.dataset.datasetbase import DataSetBase
from lib.utils.util import unpack_file, check_path


__all__ = ['DukeMTMCVidReID']


class DukeMTMCVidReID(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('DukeMTMCVidReID', split_id, 'db', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'DukeMTMC-VideoReID.zip'

        self.raw_data_folder = self.store_dir / 'DukeMTMC-VideoReID'
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'

        self.train_dir = self.raw_data_folder / 'train'
        self.query_dir = self.raw_data_folder / 'query'
        self.gallery_dir = self.raw_data_folder / 'gallery'

        self.resize_hw = (256, 128)
        self.init()

    def check_raw_file(self):
        if not self.zipfiles_dir.exists():
            check_path(self.zipfiles_dir.parent, create=True)
            urlretrieve(self.dataset_url, self.zipfiles_dir)
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)

    def _get_dict(self):

        train_images = self._process_dir(self.train_dir)
        probe_images = self._process_dir(self.query_dir)
        gallery_images = self._process_dir(self.gallery_dir)

        img_list, train_info, probe_info, gallery_info = self._process_split_info(train_images, probe_images, gallery_images)

        train_id = np.unique(train_info[:, 0])
        probe_id = np.unique(probe_info[:, 0])
        gallery_id = np.unique(gallery_info[:, 0])
        assert np.intersect1d(train_id, probe_id).size == 0
        assert np.intersect1d(probe_id, gallery_id).size == probe_id.size
        assert gallery_info[-1, 3] == len(img_list)

        data_dict = {}
        data_dict['dir'] = img_list

        data_split = {}
        data_split['train'] = train_info
        data_split['probe'] = probe_info
        data_split['gallery'] = gallery_info
        data_split['info'] = 'DukeMTMC-VideoReID. Split ID {:2d}.'.format(0)

        data_dict['split'] = [data_split]
        data_dict['info'] = 'DukeMTMC-VideoReID Dataset. One split.'

        return data_dict

    def _process_split_info(self, train_images, probe_images, gallery_images):
        img_list = []
        idx = 0
        train_info = []
        for train_track_i in train_images:
            track_len = len(train_track_i[0])
            img_list.extend(train_track_i[0])
            train_info.append([train_track_i[1], train_track_i[2], idx, idx+track_len, track_len])
            idx += track_len
        train_info = np.asarray(train_info, dtype=np.int64)

        gallery_info = []
        for gallery_track_i in gallery_images:
            track_len = len(gallery_track_i[0])
            img_list.extend(gallery_track_i[0])
            gallery_info.append([gallery_track_i[1], gallery_track_i[2], idx, idx+track_len, track_len])
            idx += track_len
        gallery_info = np.asarray(gallery_info, dtype=np.int64)

        probe_info = []
        for probe_track_i in probe_images:
            base_img_name = [osp.basename(img_dir) for img_dir in probe_track_i[0]]
            track_len = len(probe_track_i[0])
            finded_track_info = None
            for gallery_track_i in gallery_images:
                if finded_track_info is not None:
                    break
                if gallery_track_i[1] == probe_track_i[1] and gallery_track_i[2] == probe_track_i[2]:
                    for gallery_img_i in gallery_track_i[0]:
                        if base_img_name[0] in gallery_img_i:
                            finded_track_info = gallery_track_i[0]
                            break
                else:
                    continue
            if finded_track_info is None:
                raise KeyError
            new_img_idx_list = []
            for base_img_name_i in base_img_name:
                for gallery_img_i in finded_track_info:
                    if base_img_name_i in gallery_img_i:
                        img_idx = img_list.index(gallery_img_i)
                        if len(new_img_idx_list) > 0:
                            assert img_idx == new_img_idx_list[-1]+1
                        new_img_idx_list.append(img_idx)
                        break
            assert len(new_img_idx_list) == track_len

            track_begin_idx = new_img_idx_list[0]
            probe_info.append([probe_track_i[1], probe_track_i[2], track_begin_idx, track_begin_idx+track_len, track_len])
        probe_info = np.asarray(probe_info, dtype=np.int64)

        return img_list, train_info, probe_info, gallery_info

    def _process_dir(self, data_dir):
        pdirs = glob.glob(osp.join(data_dir, '*'))  # avoid .DS_Store
        pdirs.sort()
        self.logger.info("Processing '{}' with {} person identities".format(data_dir, len(pdirs)))

        tracklets = []

        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            tdirs = glob.glob(osp.join(pdir, '*'))
            tdirs.sort()

            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                raw_img_paths.sort()
                num_imgs = len(raw_img_paths)

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
        return tracklets
