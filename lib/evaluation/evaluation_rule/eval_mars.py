import numpy as np
from .eval_base import EvaluatorBase

'''
The evaluation code on MARS dataset is the python implementation of the evaluation code from https://github.com/liangzheng06/MARS-evaluation
'''

def compute_AP(good_index, junk_index, order):
    cmc = np.zeros(order.size, dtype=np.float32)
    nGood = good_index.size
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    j = 0
    good_now = 0
    nJunk = 0
    for i_order, order_i in enumerate(order):
        flag = False
        if good_index[good_index == order_i].size != 0:
            cmc[i_order - nJunk:] = 1
            flag = True
            good_now += 1
        if junk_index[junk_index == order_i].size != 0:
            nJunk += 1
            continue

        if flag:
            intersect_size += 1.0
        recall = intersect_size / nGood
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j += 1
        if good_now == nGood:
            break
    return ap, cmc


def compute_AP_multiCam(good_index, junk_index, order, cam_q, cam_g, nCam):
    good_cam = cam_g[good_index]
    good_cam_uni = np.unique(good_cam)
    ap_multi = np.zeros(nCam,dtype=np.float32)

    good_cam_now = cam_q
    nGood = junk_index.size - 1
    junk_index_now = np.concatenate((good_index, np.array([order[0]])))
    good_index_now = np.setdiff1d(junk_index, np.array([order[0]]))

    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    j = 0
    good_now = 0

    for i_order, order_i in enumerate(order):
        flag = False
        if good_index_now[good_index_now == order_i].size != 0:
            flag = True
            good_now += 1
        if junk_index_now[junk_index_now == order_i].size != 0:
            continue
        if flag:
            intersect_size += 1
        if nGood == 0:
            ap_multi[good_cam_now] = 0
            break

        recall = intersect_size/nGood
        precision = intersect_size/(j+1)
        ap = ap + (recall - old_recall) * ((old_precision+precision)/2)
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == nGood:
            ap_multi[good_cam_now] = ap
            break

    for i_cam, good_cam_now in enumerate(good_cam_uni):
        nGood = good_cam[good_cam == good_cam_now].size
        pos_junk = np.where(good_cam != good_cam_now)
        junk_index_now = np.concatenate((junk_index,good_index[pos_junk]), axis=0)
        pos_good = np.where(good_cam == good_cam_now)
        good_index_now = good_index[pos_good]
        old_recall = 0
        old_precision = 1.0
        ap = 0.0
        intersect_size = 0.0
        j = 0
        good_now = 0
        for i_order, order_i in enumerate(order):
            flag = False
            if good_index_now[good_index_now == order_i].size != 0:
                flag = True
                good_now += 1
            if junk_index_now[junk_index_now == order_i].size != 0:
                continue
            if flag:
                intersect_size += 1
            recall = intersect_size/nGood
            precision = intersect_size/(j+1)
            ap = ap + (recall - old_recall) * ((old_precision + precision)/2)
            old_recall = recall
            old_precision = precision
            j += 1
            if good_now == nGood:
                ap_multi[good_cam_now] = ap
                break
    return ap_multi


def compute_r1_multiCam(good_index, junk_index, order, cam_q, cam_g, nCam):
    good_cam = cam_g[good_index]
    good_cam_uni = np.unique(good_cam)
    r1 = np.ones(nCam, dtype=np.float32) - 3

    good_cam_now = cam_q
    nGood = junk_index.size - 1
    junk_index_now = np.concatenate((good_index, np.array([order[0]])))
    good_index_now = np.setdiff1d(junk_index, np.array([order[0]]))
    good_now = 0
    for i_order, order_i in enumerate(order):
        flag = False
        if nGood == 0:
            r1[good_cam_now] = -1
            break

        if good_index_now[good_index_now == order_i].size != 0:
            flag = True
            good_now += 1
        if junk_index_now[junk_index_now == order_i].size != 0:
            continue
        if not flag:
            r1[good_cam_now] = 0
            break
        if flag:
            r1[good_cam_now] = 1
            break

    for i_cam, good_cam_now in enumerate(good_cam_uni):
        nGood = good_cam[good_cam == good_cam_now].size
        pos_junk = np.where(good_cam != good_cam_now)
        junk_index_now = np.concatenate((junk_index, good_index[pos_junk]))
        pos_good = np.where(good_cam == good_cam_now)
        good_index_now = good_index[pos_good]
        for i_order, order_i in enumerate(order):
            flag = False
            if nGood == 0:
                r1[good_cam_now] = -1
                break
            if good_index_now[good_index_now == order_i].size != 0:
                flag = True
            if junk_index_now[junk_index_now == order_i].size != 0:
                continue

            if not flag:
                r1[good_cam_now] = 0
                break
            if flag:
                r1[good_cam_now] = 1
                break

    return r1


def confusion_matrix(ap, r1, cam_p, nCam):
    ap_mat = np.zeros((nCam, nCam), np.float32)
    r1_mat = np.zeros((nCam, nCam), np.float32)
    count1 = np.zeros((nCam, nCam), np.float32) + 1e-5
    count2 = np.zeros((nCam, nCam), np.float32) + 1e-5

    for i_p, p_i in enumerate(cam_p):
        for cam_i in range(nCam):
            ap_mat[p_i, cam_i] += ap[i_p, cam_i]
            if ap[i_p, cam_i] != 0:
                count1[p_i, cam_i] += 1
            if r1[i_p, cam_i] >= 0:
                r1_mat[p_i, cam_i] += r1[i_p, cam_i]
                count2[p_i, cam_i] += 1

    ap_mat = np.true_divide(ap_mat, count1)
    r1_mat = np.true_divide(r1_mat, count2)
    return r1_mat, ap_mat


class EvalMARS(EvaluatorBase):
    def _map_multi_cam(self, distMat):
        assert self.nCam == 6
        ap = np.zeros(self.probe_num, np.float32)
        cmc = np.zeros((self.probe_num, self.gallery_num), np.float32)
        # r1_pairwise = np.zeros((self.probe_num, self.nCam), np.float32)
        # ap_pairwise = np.zeros((self.probe_num, self.nCam), np.float32)
        for i_p in range(self.probe_num):
            dist = distMat[i_p, ...]
            p_id = self.probe_id[i_p]
            cam_p = self.probe_cam_id[i_p]
            # cam_g = self.gallery_cam_id.copy()
            pos = np.where(self.gallery_id == p_id)[0]
            pos2 = np.where(self.gallery_cam_id[pos] != cam_p)
            good_index = pos[pos2]
            pos3 = np.where(self.gallery_cam_id[pos] == cam_p)
            temp_junk_index = pos[pos3]
            junk_index = temp_junk_index if self.junk_index is None else np.concatenate(
                (self.junk_index, temp_junk_index), axis=0)
            dist_order = np.argsort(dist)
            ap[i_p, ...], cmc[i_p, ...] = compute_AP(good_index, junk_index, dist_order)
            # ap_pairwise[i_p, ...] = compute_AP_multiCam(good_index, junk_index, dist_order, cam_p, cam_g, self.nCam)
            # r1_pairwise[i_p, ...] = compute_r1_multiCam(good_index, junk_index, dist_order, cam_p, cam_g, self.nCam)

        cmc = np.sum(cmc, axis=0) / self.probe_num
        mAP = np.sum(ap) / ap.size

        # r1_mat, ap_mat = confusion_matrix(ap_pairwise, r1_pairwise, self.probe_cam_id.copy(), self.nCam)
        # r1_mat = r1_mat * 100.0
        # ap_mat = ap_mat * 100.0
        # print('R1 MAP:\n%s\nAP_MAP:\n%s' % (r1_mat, ap_mat))

        return cmc, mAP

    def _get_cmc_mAP(self, distMat):
        cmc, mAP = self._map_multi_cam(distMat.copy())
        return cmc, mAP