from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
from functools import wraps
from datetime import datetime, timezone, timedelta


def time_type_transform(allTime):
    day = 24 * 60 * 60
    hour = 60 * 60
    min = 60
    if allTime < 60:
        return "%d sec" % math.ceil(allTime)
    elif allTime > day:
        days = divmod(allTime, day)
        return "%d days, %s" % (int(days[0]), time_type_transform(days[1]))
    elif allTime > hour:
        hours = divmod(allTime, hour)
        return '%d hours, %s' % (int(hours[0]), time_type_transform(hours[1]))
    else:
        mins = divmod(allTime, min)
        return "%d mins, %d sec" % (int(mins[0]), math.ceil(mins[1]))


def timer_run_time(func):
    @wraps(func)
    def _timer_run_time(*args, **kwargs):
        start_time = datetime.utcnow()
        ret = func(*args, **kwargs)
        end_time = datetime.utcnow()
        duration = end_time - start_time
        print('Duration :', duration.total_seconds(), 's')
        return ret
    return _timer_run_time


def timer_lite(func):
    start_time = datetime.utcnow()
    func()
    end_time = datetime.utcnow()
    duration = end_time - start_time
    duration = time_type_transform(duration.total_seconds())
    return duration


def get_unified_time(d=None, offset=8):
    if d is not None:
        utc_dt = d.utcnow()
    else:
        utc_dt = datetime.utcnow()
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=offset)))
    return bj_dt


def time_tag():
    return str(get_unified_time().strftime('%Y-%m-%d_%H%M%S'))


class TimerClock(object):
    def __init__(self,parts=1):
        assert parts != 0
        self.parts = parts
        self._time = time.time()
        self._state_time = self._time

    def Timing(self):
        time_now = time.time()
        duration = time_now - self._state_time
        self._state_time = time_now
        return duration/self.parts

    def printf(self):
        print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


class AverageMeter(object):
    def __init__(self, n):
        self.len = n
        self._value_vector = np.zeros(self.len, dtype=np.float32)
        self.count = 0

    def updata(self, value):
        self._value_vector += np.asarray(value, dtype=self._value_vector.dtype)
        self.count += 1

    def reset(self):
        self._value_vector = np.zeros(self.len, dtype=np.float32)
        self.count = 0

    def __getitem__(self, item):
        return self._value_vector[item] / self.count

    def __len__(self):
        return self.len


class TensorBuffer(object):
    def __init__(self, block_size_list, func):
        self.block_size_list = block_size_list
        self.block_size_max = max(self.block_size_list)
        self._func = func

        if self.block_size_max == 1:
            self.push = self.push_image
        else:
            self.push = self.push_video
            self.idx = 0
            self.fea_shape = None

        self.result = []
        self.block_idx = 0
        self.is_end = False

    def push_image(self, fea):
        batch_size = fea.size(0)
        for i_fea in range(batch_size):
            self.result.append(fea[i_fea])
            self.block_idx += 1
            if self.block_idx == len(self.block_size_list):
                self.is_end = True

    def push_video(self, fea):
        fea_shape = fea.size()
        batch_size = fea_shape[0]
        if self.fea_shape is None:
            fea_shape = fea_shape[1:]
            self.fea_shape = fea_shape
            self.feaMat = torch.from_numpy(np.zeros((self.block_size_max,) + fea_shape, np.float32)).cuda()

        read_range = [0, 0]
        store_range = [self.idx, self.idx]
        for i_batch in range(batch_size):
            read_range[1] += 1
            store_range[1] += 1
            self.idx += 1
            if self.idx == self.block_size_list[self.block_idx]:
                self.feaMat[store_range[0]:store_range[1]] = fea[read_range[0]:read_range[1], ...]
                self.result.append(self._func(self.feaMat[:self.block_size_list[self.block_idx]]))

                self.idx = 0
                read_range[0] = read_range[1]
                store_range = [0, 0]
                self.block_idx += 1
                if self.block_idx == len(self.block_size_list):
                    self.is_end = True

        if read_range[0] != read_range[1]:
            self.feaMat[store_range[0]:store_range[1]] = fea[read_range[0]:read_range[1], ...]