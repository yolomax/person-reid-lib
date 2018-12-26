import os
import os.path as osp
import sys
import numpy as np
from lib.utils.util import DataPacker
from pathlib import Path


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.split(os.path.realpath(__file__))[0]

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..')
add_path(lib_path)

model_dir = Path(str(this_dir)).parent / 'tasks'
task_list = ['task1', 'task2', 'task3', 'task4', 'task5']   # the name of the task folder
epoch_list = None
cmc = []
mAP = []
for task_i in task_list:
    cmc_dir = model_dir / str(str(task_i) + '/output/result')
    cmc_dir_tmp = cmc_dir.glob('cmc_*.json')
    cmc_dir_tmp = sorted([x.name for x in cmc_dir_tmp])[-1]
    print(cmc_dir_tmp)
    cmc_dir = cmc_dir / cmc_dir_tmp
    cmc_i = DataPacker.load(cmc_dir)
    solver_list = list(cmc_i.keys())
    if epoch_list is None:
        epoch_list = list(cmc_i[solver_list[0]].keys())
        epoch_list = [int(x) for x in epoch_list]
        epoch_list.sort()
        epoch_list = [str(x) for x in epoch_list]
    for solver_id in solver_list:
        epoch_list_i = list(cmc_i[solver_list[0]].keys())
        epoch_list_i = [int(x) for x in epoch_list_i]
        epoch_list_i.sort()
        epoch_list_i = [str(x) for x in epoch_list_i]
        assert epoch_list_i == epoch_list
        cmc_tmp = []
        mAP_tmp = []
        for epoch_id in epoch_list:
            cmc_tmp.append(cmc_i[solver_id][epoch_id]['cmc'])
            mAP_tmp.append(cmc_i[solver_id][epoch_id]['mAP'])
        cmc.append(cmc_tmp)
        mAP.append(mAP_tmp)

cmc = np.asarray(cmc, dtype=np.float32)
mAP = np.asarray(mAP, dtype=np.float32)
# for i in range(cmc.shape[0]):
#     print(i)
#     print(mAP[i], cmc[i])

print(cmc.shape, mAP.shape)
print('CMC and mAP for %8d times.' % cmc.shape[0])
cmc = np.mean(cmc, axis=0)
mAP = mAP.mean(axis=0)
print('Epoch      mAP     R1     R5      R10     R20  ')

for i, epoch_id in enumerate(epoch_list):
    print(' %-8d%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f' % (int(epoch_id), mAP[i], cmc[i][0], cmc[i][1], cmc[i][2], cmc[i][3]))
