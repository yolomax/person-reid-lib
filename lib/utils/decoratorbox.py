from functools import wraps
import numpy as np


def release_resource(func):
    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        result = func(cls, *args, **kwargs)
        pass   # cls.xxx.close()
        return result
    return wrapper


def print_info(func):
    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        cls.logger.info('Solver information!')
        cls.logger.info('Train nPerson %d' % cls.Data.dataset.train_person_num)
        cls.logger.info('Seed {}'.format(cls.manager.seed))
        result = func(cls, *args, **kwargs)
        return result
    return wrapper


def vision_performance(func):
    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        result = func(cls, *args, **kwargs)
        cls.recorder.visual.plot('Loss', 'AvgSame', np.asarray([cls.epoch]), np.asarray([result[2]]))
        cls.recorder.visual.plot('Loss', 'AvgDiff', np.asarray([cls.epoch]), np.asarray([result[3]]))
        cls.recorder.visual.plot('Performance', 'Rank1', np.asarray([cls.epoch]), np.asarray([result[0][0]]))
        cls.recorder.visual.plot('Performance', 'Rank5', np.asarray([cls.epoch]), np.asarray([result[0][4]]))
        cls.recorder.visual.plot('Performance', 'Rank10', np.asarray([cls.epoch]), np.asarray([result[0][9]]))
        cls.recorder.visual.plot('Performance', 'Rank20', np.asarray([cls.epoch]), np.asarray([result[0][19]]))
        cls.recorder.visual.plot('Performance', 'Map', np.array([cls.epoch]), np.array([result[1]]))

        return result
    return wrapper

