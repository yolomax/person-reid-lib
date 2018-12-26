from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from .eval_base import EvaluatorBase


try:
    from .eval_lib.cython_eval import eval_market1501_wrap
    CYTHON_EVAL_AVAI = True
    print("Cython evaluation is AVAILABLE")
except ImportError:
    CYTHON_EVAL_AVAI = False
    print("Warning: Cython evaluation is UNAVAILABLE")


def compute_standard_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, logger):
    """Evaluation with market1501 metric
                Key: for each query identity, its gallery images from the same camera view are discarded.
                """
    num_q, num_g = distmat.shape
    assert num_q == q_pids.size and num_g == g_pids.size
    assert num_q == q_camids.size and num_g == g_camids.size
    if num_g < max_rank:
        max_rank = num_g
        logger.warn("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class EvalStandard(EvaluatorBase):
    def _get_cmc_mAP(self, distMat):
        if CYTHON_EVAL_AVAI:
            cmc, mAP = eval_market1501_wrap(distMat, self.probe_id, self.gallery_id, self.probe_cam_id, self.gallery_cam_id, max_rank=20)
        else:
            cmc, mAP = compute_standard_cmc_map(distMat, self.probe_id, self.gallery_id, self.probe_cam_id, self.gallery_cam_id, max_rank=20, logger=self.logger)
        return cmc, mAP