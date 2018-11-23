from __future__ import absolute_import

import sys
import os
import os.path as osp
import logging

from lib.utils.util import check_path
from lib.utils.meter import get_unified_time


class OSLogger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            check_path(osp.dirname(fpath), create=True)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class InfoLogger(object):
    def __init__(self,file_path, file_name, tag):
        self.file_path = file_path
        self.file_name = file_name
        self.tag = tag
        self.init()
        self.info()

    def info(self):
        self.logger.info('Time Tag: %s' % self.tag)

    def init(self):
        self._create(self.file_path, self.file_name)

    def _create(self, file_path, file_name, leval_all=logging.DEBUG, level_stream=logging.INFO, level_file=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(leval_all)
        logfile = file_path / file_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(level_file)

        ch = logging.StreamHandler()
        ch.setLevel(level_stream)

        converter = lambda x, y: get_unified_time().timetuple()
        logging.Formatter.converter = converter

        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger

    def finish(self):
        pass