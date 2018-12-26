from lib.utils.util import ConstType, check_path
from lib.recorder.visual import Visual
from lib.recorder.logger import OSLogger, InfoLogger
import sys


class Recorder(ConstType):

    def __init__(self, task_dir, time_tag, device):
        self.father_path = task_dir
        self.time_tag = time_tag
        self.logger = InfoLogger(check_path(self.father_path / 'output/log', create=False), 'log.txt',
                                 self.time_tag).logger
        self.visual = Visual('Visdom', self.father_path, self.time_tag, device, self.logger)
        # sys.stdout = OSLogger(self.father_path / ('log_' + self.time_tag + '.txt'))



    def exit(self):
        if hasattr(self, 'visual'):
            self.visual.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()