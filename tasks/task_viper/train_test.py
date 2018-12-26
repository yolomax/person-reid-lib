#!usr/local/bin/python

import _init_paths
import torch
import numpy as np
import random
from lib.utils.manager import Manager
from lib.utils.util import file_abs_path, ParseArgs
from lib.utils.meter import timer_lite
from solver import Solver
from lib.evaluation.eval_tools import compute_rank


def main():
    cur_dir = file_abs_path(__file__)
    manager = Manager(cur_dir, seed=None, mode='Train')
    logger = manager.logger
    ParseArgs(logger)
    if manager.seed is not None:
        random.seed(manager.seed)
        np.random.seed(manager.seed)
        torch.manual_seed(manager.seed)

    # ['iLIDS-VID', 'PRID-2011', 'LPW', 'MARS', 'VIPeR', 'Market1501', 'CUHK03', 'CUHK01', 'DukeMTMCreID', 'GRID', 'DukeMTMC-VideoReID']
    #       0            1         2      3        4          5           6         7             8           9             10

    manager.set_dataset(4)

    perf_box = {}
    repeat_times = 10
    for task_i in range(repeat_times):
        manager.split_id = int(task_i) 
        task = Solver(manager)
        train_test_time = timer_lite(task.run)
        perf_box[str(task_i)] = task.perf_box
        manager.store_performance(perf_box)

        logger.info('-----------Total time------------')
        logger.info('Split ID:' + str(task_i) + '  ' + str(train_test_time))
        logger.info('---------------------------------')

    compute_rank(perf_box, logger)


if __name__ == '__main__':
    main()
