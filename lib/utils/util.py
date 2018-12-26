from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import shutil
import argparse
import os
import errno
import pickle
import numpy as np
from pathlib import Path


def empty_folder(folder_dir):
    folder_dir = Path(folder_dir)
    assert folder_dir.is_dir()
    if folder_dir.exists():
        shutil.rmtree(folder_dir)
        folder_dir.mkdir(exist_ok=True)


def remove_folder(folder_dir):
    folder_dir = Path(folder_dir)
    assert folder_dir.is_dir()
    if folder_dir.exists():
        shutil.rmtree(folder_dir)


def file_abs_path(arg):
    return Path(os.path.realpath(arg)).parent


def remove_file(file_path):
    file_path = Path(file_path)
    assert file_path.exists() and file_path.is_file()
    os.remove(file_path)


def check_path(folder_dir, create=False):
    folder_dir = Path(folder_dir)
    if not folder_dir.exists():

        if create:
            try:
                os.makedirs(folder_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        else:
            raise IOError
    return folder_dir


def copy_file_to(source_dir, target_dir):

    if os.path.isfile(source_dir):
        check_path(target_dir, create=True)
        shutil.copy(source_dir, target_dir)
    else:
        raise FileExistsError


def unpack_file(file_path, target_path, logger=None):
    if logger is None:
        printf = print
    else:
        printf = logger.info

    file_path = Path(file_path)
    assert file_path.exists() and file_path.is_file()
    assert os.path.exists(target_path) and os.path.isdir(target_path)
    printf('Begin extract file from ' + str(file_path) + ' to ' + str(target_path))
    shutil.unpack_archive(str(file_path), target_path)
    printf('Extract Finish')


def np_filter(arr, *arg):
    temp_arr = arr
    for i_axis, axis_i in enumerate(arg):
        map_list = []
        for i_elem, elem_i in enumerate(axis_i):
            temp_elem_arr = temp_arr[temp_arr[:, i_axis] == elem_i]
            map_list.append(temp_elem_arr)
        temp_arr = np.concatenate(map_list, axis=0)
    return temp_arr


class ConstType(object):
    class ConstError(TypeError):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError
        else:
            self.__dict__[key] = value


class ParseArgs(object):
    def __init__(self, logger=None):
        parser = argparse.ArgumentParser(description='ReID')
        parser.add_argument('--gpu', default='all', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
        self.args = parser.parse_args()

        if logger is None:
            printf = print
        else:
            printf = logger.info

        if self.args.gpu != 'all':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
            printf('Choose GPU ID: ' + self.args.gpu)


class DataPacker(object):
    @staticmethod
    def dump(info, file_path, logger=None):
        if logger is None:
            printf = print
        else:
            printf = logger.info
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'wb') as f:
            pickle.dump(info, f)
        printf('Store data ---> ' + str(file_path))

    @staticmethod
    def load(file_path, logger=None):
        if logger is None:
            printf = print
        else:
            printf = logger.info
        check_path(file_path)
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
            printf('Load data <--- ' + str(file_path))
            return info
