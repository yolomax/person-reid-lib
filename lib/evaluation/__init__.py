from lib.evaluation.evaluation_rule.eval_base import EvaluatorBase
from lib.evaluation.evaluation_rule.eval_standard import EvalStandard
from lib.evaluation.evaluation_rule.eval_mars import EvalMARS
from lib.evaluation.evaluation_rule.eval_cuhk03 import EvalCUHK03

__all__ = ['Evaluator']


class Evaluator(object):
    def __init__(self, store_search_result, name, task_dir, cuhk03_classic=False, logger=None):
        self.dataset_name = name
        self.store_search_result = store_search_result
        self.task_dir = task_dir
        self.logger = logger

        self._eval_factory = {'PRID-2011': EvalStandard,
                              'iLIDS-VID': EvalStandard,
                              'MARS': EvalMARS,
                              'LPW': EvalStandard,
                              'CUHK01': EvalStandard,
                              'CUHK03': EvalStandard,
                              'DukeMTMCreID': EvalStandard,
                              'GRID': EvalStandard,
                              'Market1501': EvalStandard,
                              'VIPeR': EvalStandard,
                              'DukeMTMC-VideoReID': EvalStandard}

        if self.dataset_name not in self._eval_factory:
            raise KeyError
        if self.dataset_name == 'CUHK03' and cuhk03_classic:
            self.logger.info('CUHK03 Classic split')
            self._eval_rule = EvalCUHK03
        else:
            self._eval_rule = self._eval_factory[self.dataset_name]
        self.logger.info('{0:-^60}'.format('Choose eval rule ' + self._eval_rule.__name__))

        self._false_example = []
        self._right_example = []
        self._eval_manager = None

    def _example_buffer(self, examlple):
        false_example, right_examlple = examlple
        self._false_example.append(false_example)
        self._right_example.append(right_examlple)

    def store_example(self):
        EvaluatorBase.store_search_example(self.task_dir, self._false_example,self._right_example)

    def reset(self, *args, **kwargs):
        self._eval_manager = self._eval_rule(*args, **kwargs, logger=self.logger)

    def set_feature_buffer(self, func=None):
        self._eval_manager.set_feature_buffer(func)

    def count(self, *args, **kwargs):
        self._eval_manager.count(*args, **kwargs)

    def final_result(self):
        cmc, mAP, avgSame, avgDiff, distMat = self._eval_manager.final_result()
        if self.store_search_result:
            self._example_buffer(self._eval_manager.false_positive(distMat))
        return cmc, mAP, avgSame, avgDiff