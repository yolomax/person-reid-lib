from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import h5py
from PIL import Image
import os.path as osp
from scipy.io import loadmat
from urllib.request import urlretrieve
from .datasetbase import DataSetBase
from lib.utils.util import unpack_file, copy_file_to, remove_folder, check_path, DataPacker


__all__ = ['CUHK03']


class CUHK03(DataSetBase):
    def __init__(self, root_dir, rawfiles_dir, split_id, npr=None, logger=None):
        super().__init__('CUHK03', split_id, 'h5', root_dir, logger)
        self.zipfiles_dir = rawfiles_dir / 'cuhk03_release.zip'

        self.raw_data_folder = self.store_dir / 'cuhk03_release'
        self.raw_mat_path = self.raw_data_folder / 'cuhk-03.mat'
        self.split_config_path = self.store_dir / 'split_config'

        self.imgs_detected_dir = self.store_dir / 'images_detected'
        self.imgs_labeled_dir = self.store_dir / 'images_labeled'

        self.split_classic_det_json_path = self.split_config_path / 'splits_classic_detected.json'
        self.split_classic_lab_json_path = self.split_config_path / 'splits_classic_labeled.json'

        self.split_new_det_json_path = self.split_config_path / 'splits_new_detected.json'
        self.split_new_lab_json_path = self.split_config_path / 'splits_new_labeled.json'

        self.split_new_mat_url = 'https://github.com/zhunzhong07/person-re-ranking/archive/master.zip'
        self.split_new_det_mat_path = self.store_dir / 'cuhk03_new_protocol_config_detected.mat'
        self.split_new_lab_mat_path = self.store_dir / 'cuhk03_new_protocol_config_labeled.mat'

        self.resize_hw = (256, 128)
        self.init()

    def check_raw_file(self):
        assert self.zipfiles_dir.exists()
        if not self.raw_data_folder.exists():
            unpack_file(self.zipfiles_dir, self.store_dir, self.logger)

        check_path(self.split_config_path, create=True)

        if not self.split_new_det_mat_path.exists() or not self.split_new_lab_mat_path.exists():
            config_file_dir = self.split_config_path / 'person-re-ranking-master'
            if not config_file_dir.exists():
                config_file_path = self.store_dir / 'person-re-ranking-master.zip'
                if not config_file_path.exists():
                    urlretrieve(self.split_new_mat_url, config_file_path)

                unpack_file(config_file_path, self.split_config_path, self.logger)

            if not self.split_new_det_mat_path.exists():
                copy_file_to(
                    self.split_config_path / 'person-re-ranking-master/evaluation/data/CUHK03/cuhk03_new_protocol_config_detected.mat',
                    self.store_dir)
            if not self.split_new_lab_mat_path.exists():
                copy_file_to(
                    self.split_config_path / 'person-re-ranking-master/evaluation/data/CUHK03/cuhk03_new_protocol_config_labeled.mat',
                    self.store_dir)

            if config_file_dir.exists():
                remove_folder(config_file_dir)

        assert self.raw_mat_path.exists()

    def _get_dict(self):
        self._preprocess()

        split_paths = [self.split_new_det_json_path,
                       self.split_new_lab_json_path,
                       self.split_classic_det_json_path,
                       self.split_classic_lab_json_path]

        split_paths_info = ['New detected', 'New labeled', 'Classic detected', 'Classic labeled']

        images_list = set()
        for split_path in split_paths:
            splits = DataPacker.load(split_path, self.logger)
            for split in splits:
                train = split['train']
                query = split['query']
                gallery = split['gallery']
                for data_tmp in [train, query, gallery]:
                    for img_info_tmp in data_tmp:
                        images_list.add(img_info_tmp[0])
        images_list = [img_dir for img_dir in images_list]

        data_splits = []
        true_split_id = 0
        for split_path_id, split_path in enumerate(split_paths):
            splits = DataPacker.load(split_path, self.logger)
            for split_id, split in enumerate(splits):
                train = split['train']
                query = split['query']
                gallery = split['gallery']

                tqg_info = []
                for data_tmp in [train, query, gallery]:
                    data_info = []
                    for img_info_tmp in data_tmp:
                        idx = images_list.index(img_info_tmp[0])
                        data_info.append([img_info_tmp[1], img_info_tmp[2], idx, idx+1, 1])
                    tqg_info.append(data_info)
                train_images = np.asarray(tqg_info[0], np.int64)
                probe_images = np.asarray(tqg_info[1], np.int64)
                gallery_images = np.asarray(tqg_info[2], np.int64)

                assert np.unique(train_images[:, 2]).size == train_images.shape[0]
                assert np.unique(probe_images[:, 2]).size == probe_images.shape[0]
                assert np.unique(gallery_images[:, 2]).size == gallery_images.shape[0]

                probe_id = np.unique(probe_images[:, 0])
                gallery_id = np.unique(gallery_images[:, 0])
                assert np.intersect1d(probe_id, gallery_id).size == probe_id.size

                data_split = {}
                data_split['train'] = train_images
                data_split['probe'] = probe_images
                data_split['gallery'] = gallery_images
                data_split['info'] = 'CUHK03 dataset. {}. Split ID {:2d}.'.format(split_paths_info[split_path_id], true_split_id)
                true_split_id += 1
                data_splits.append(data_split)

        data_dict = {}
        data_dict['dir'] = images_list
        data_dict['split'] = data_splits
        data_dict['info'] = 'CUHK03 Dataset. \nSplit ID: 0 New detected\nSplit ID: 1 New labeled\nSplit ID: 2-21 Classic detected\nSplit ID: 22-41 Classic labeled\n'

        return data_dict

    def _preprocess(self):
        """
        This function is a bit complex and ugly, what it does is
        1. Extract data from cuhk-03.mat and save as png images.
        2. Create 20 classic splits. (Li et al. CVPR'14)
        3. Create new split. (Zhong et al. CVPR'17)
        """
        self.logger.info(
            "Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")
        if self.imgs_labeled_dir.exists() and \
                self.imgs_detected_dir.exists() and \
                self.split_classic_det_json_path.exists() and \
                self.split_classic_lab_json_path.exists() and \
                self.split_new_det_json_path.exists() and \
                self.split_new_lab_json_path.exists():
            return

        check_path(self.imgs_detected_dir, create=True)
        check_path(self.imgs_labeled_dir, create=True)

        self.logger.info("Extract image data from {} and save as png".format(self.raw_mat_path))
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = []  # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                # skip empty cell
                if img.size == 0 or img.ndim < 3: continue
                img = Image.fromarray(img, mode='RGB')

                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(campid + 1, pid + 1, viewid, imgid + 1)
                img_path = osp.join(save_dir, img_name)
                img.save(img_path)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(name):
            self.logger.info("Processing {} images (extract and save) ...".format(name))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if name == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[name][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(camp[pid, :], campid, pid, imgs_dir)
                    assert len(img_paths) > 0, "campid{}-pid{} has no images".format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                self.logger.info("done camera pair {} with {} identities".format(campid + 1, num_pids))
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        self.logger.info("Creating classic splits (# = 20) ...")
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

        DataPacker.dump(splits_classic_det, self.split_classic_det_json_path, self.logger)
        DataPacker.dump(splits_classic_lab, self.split_classic_lab_json_path, self.logger)
        mat.close()

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2])
                pid = pids[idx]
                if relabel: pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(filelist, pids, pid2label, train_idxs, img_dir, relabel=True)
            query_info = _extract_set(filelist, pids, pid2label, query_idxs, img_dir, relabel=False)
            gallery_info = _extract_set(filelist, pids, pid2label, gallery_idxs, img_dir, relabel=False)
            return train_info, query_info, gallery_info

        self.logger.info("Creating new splits for detected images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path),
            self.imgs_detected_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        DataPacker.dump(splits, self.split_new_det_json_path)

        self.logger.info("Creating new splits for labeled images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path),
            self.imgs_labeled_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        DataPacker.dump(splits, self.split_new_lab_json_path)
