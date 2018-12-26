import torch
import numpy as np
import h5py
from copy import deepcopy
from lib.evaluation.distance_tool import Distance
from lib.utils.util import check_path
from lib.evaluation.evaluation_rule.utils import TensorBuffer


class EvaluatorBase(object):
    def __init__(self, dataset, distance_func='L2Euclidean', logger=None):
        self.logger = logger
        self._distance = Distance(distance_func)
        self.img_dir = deepcopy(dataset.images_dir_list)

        self.test_info = deepcopy(dataset.test_info)
        self.probe_index = deepcopy(dataset.probe_index)
        self.gallery_index = deepcopy(dataset.gallery_index)
        self.junk_index = deepcopy(dataset.junk_index)
        self.nCam = dataset.test_cam_num

        self.probe_id = self.test_info[self.probe_index, 0]
        self.probe_cam_id = self.test_info[self.probe_index, 1]
        self.gallery_id = self.test_info[self.gallery_index, 0]
        self.gallery_cam_id = self.test_info[self.gallery_index, 1]

        self.test_num = self.test_info.shape[0]
        self.probe_num = self.probe_index.size
        self.gallery_num = self.gallery_index.size

        self.distMat = torch.from_numpy(np.zeros((self.probe_num, self.gallery_num), np.float32)).cuda()
        self.avgSame = torch.zeros(1).cuda()
        self.avgDiff = torch.zeros(1).cuda()
        self.avgSameCount = torch.zeros(1).cuda()
        self.avgDiffCount = torch.zeros(1).cuda()

        self.test_idx = 0
        self.fea_shape = None
        self.probe_dst_max = max(1, 12180*2 // self.gallery_num)

    def set_feature_buffer(self, func):
        self.feature_buffer = TensorBuffer(self.test_info[:, 4].tolist(), func)

    def _feature_distance(self, feaMat):
        probe_feature = torch.index_select(feaMat, dim=0, index=torch.from_numpy(self.probe_index).long().cuda())
        gallery_feature = torch.index_select(feaMat, dim=0, index=torch.from_numpy(self.gallery_index).long().cuda())

        idx = 0
        while idx + self.probe_dst_max < self.probe_num:
            tmp_probe_fea = probe_feature[idx:idx+self.probe_dst_max]
            dst_pg = self._feature_distance_mini(tmp_probe_fea, gallery_feature)
            self.distMat[idx:idx+self.probe_dst_max] += dst_pg
            idx += self.probe_dst_max
        tmp_probe_fea = probe_feature[idx:self.probe_num]
        dst_pg = self._feature_distance_mini(tmp_probe_fea, gallery_feature)
        self.distMat[idx:self.probe_num] += dst_pg

        for i_p, p in enumerate(self.probe_index):
            for i_g, g in enumerate(self.gallery_index):
                if self.test_info[p, 0] != self.test_info[g, 0]:
                    self.avgDiff = self.avgDiff + self.distMat[i_p, i_g]
                    self.avgDiffCount = self.avgDiffCount + 1
                elif p != g:
                    self.avgSame = self.avgSame + self.distMat[i_p, i_g]
                    self.avgSameCount = self.avgSameCount + 1

    def _feature_distance_mini(self, probe_fea, gallery_fea):
        probe_num = probe_fea.size(0)
        fea_shape = probe_fea.size()[1:]
        gallery_num = gallery_fea.size(0)
        probe_fea = probe_fea.view((probe_num, 1) + fea_shape)
        probe_fea = probe_fea.expand((probe_num, gallery_num) + fea_shape).contiguous()
        gallery_fea = gallery_fea.view((1, gallery_num) + fea_shape)
        gallery_fea = gallery_fea.expand((probe_num, gallery_num) + fea_shape).contiguous()
        probe_fea = probe_fea.view((probe_num*gallery_num,) + fea_shape)
        gallery_fea = gallery_fea.view((probe_num*gallery_num,) + fea_shape)
        dst = self._distance(probe_fea, gallery_fea).view(probe_num, gallery_num)
        return dst

    def count(self, args):
        self.feature_buffer.push(args)
        if self.feature_buffer.is_end:
            feaMat = torch.stack(self.feature_buffer.result, dim=0)
            self._feature_distance(feaMat)

    def _get_cmc_mAP(self, distMat):
        raise NotImplementedError

    def final_result(self):
        self.avgSame = self.avgSame / (self.avgSameCount + 1e-5)
        self.avgDiff = self.avgDiff / (self.avgDiffCount + 1e-5)
        avgSame = self.avgSame.cpu().numpy()
        avgDiff = self.avgDiff.cpu().numpy()

        distMat = self.distMat.cpu().numpy()
        cmc, mAP = self._get_cmc_mAP(distMat.copy())
        cmc = (cmc * 100.0).round(4)
        mAP = (mAP * 100.0).round(4)
        self.logger.info('mAP [%f]   Rank1 [%f] Rank5 [%f] Rank10 [%f] Rank20 [%f]' % (mAP, cmc[0], cmc[4], cmc[9], cmc[19]))
        return cmc, mAP, avgSame, avgDiff, distMat

    @staticmethod
    def store_search_example(father_path, false_example, right_example):
        file_path = check_path(father_path / 'output/log', True)
        file_dir = file_path / 'search_result.h5'
        with h5py.File(file_dir, 'w') as f:
            f['num'] = len(false_example)
            for i_false, false_i in enumerate(false_example):
                f['false_img_' + str(i_false)] = false_i[0].astype(np.string_)
                f['false_dst_' + str(i_false)] = false_i[1]
                f['false_id_' + str(i_false)] = false_i[2]
                f['right_img_' + str(i_false)] = right_example[i_false][0].astype(np.string_)
                f['right_dst_' + str(i_false)] = right_example[i_false][1]
                f['right_id_' + str(i_false)] = right_example[i_false][2]

    def false_positive(self, distMat):
        false_example = []
        false_distmat = []
        right_examlple = []
        right_distmat = []
        false_id = []
        right_id = []

        good_index_father = np.arange(self.gallery_index.size, dtype=np.int64)

        for i_p, p in enumerate(self.probe_index):
            temp_box = []
            p_info = self.test_info[p, ...]
            p_id = p_info[0]
            p_cam_id = p_info[1]

            temp_id = [p_id]
            image_dir_idx = p_info[2]
            img_dir = self.img_dir[image_dir_idx]
            temp_box.append(img_dir)

            good_index = good_index_father.copy()

            pos = np.where(self.gallery_id == p_id)[0]
            pos1 = np.where(self.gallery_cam_id[pos] == p_cam_id)
            junk_index = pos[pos1]
            good_index = np.setdiff1d(good_index, junk_index)

            good_index.sort()
            search_order = np.argsort(distMat[i_p, good_index])
            gallery_index = self.gallery_index[good_index]

            for i_s in range(10):
                idx_temp = gallery_index[search_order[i_s]]
                temp_id.append(self.test_info[idx_temp, 0])
                img_dir_idx = self.test_info[idx_temp, 2]
                img_dir = self.img_dir[img_dir_idx]
                temp_box.append(img_dir)
            temp_box = np.stack(temp_box)

            if self.test_info[gallery_index[search_order[0]], 0] != p_id:
                false_example.append(temp_box)
                false_distmat.append(distMat[i_p, good_index[search_order[:10]]])
                false_id.append(temp_id)
            else:
                right_examlple.append(temp_box)
                right_distmat.append(distMat[i_p, good_index[search_order[:10]]])
                right_id.append(temp_id)
        false_example = np.stack(false_example)
        right_examlple = np.stack(right_examlple)
        false_distmat = np.stack(false_distmat)
        right_distmat = np.stack(right_distmat)
        false_id = np.stack(false_id)
        right_id = np.stack(right_id)
        return [false_example, false_distmat, false_id], [right_examlple, right_distmat, right_id]
