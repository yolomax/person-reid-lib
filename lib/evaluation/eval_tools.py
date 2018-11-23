import numpy as np


def compute_rank(perf_box, logger):
    split_ids = list(perf_box.keys())
    epoch_ids = list(perf_box[split_ids[0]].keys())
    cmc = []
    mAP = []
    for split_id in split_ids:
        cmc_tmp = []
        mAP_tmp = []
        for epoch_id in epoch_ids:
            cmc_tmp.append(perf_box[split_id][epoch_id]['cmc'])
            mAP_tmp.append(perf_box[split_id][epoch_id]['mAP'])
        cmc.append(cmc_tmp)
        mAP.append(mAP_tmp)
    cmc = np.asarray(cmc, dtype=np.float32)
    mAP = np.asarray(mAP, dtype=np.float32)
    cmc = np.mean(cmc, axis=0)
    mAP = np.mean(mAP, axis=0)
    logger.info('CMC and mAP for %4d times.' % len(split_ids))
    logger.info('Epoch      mAP     R1     R5      R10     R20  ')
    for i, epoch_id in enumerate(epoch_ids):
        logger.info(' %-8d%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f' % (int(epoch_id),mAP[i], cmc[i][0], cmc[i][1], cmc[i][2], cmc[i][3]))