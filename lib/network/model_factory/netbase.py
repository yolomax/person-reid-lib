import torch
import numpy as np
import torch.optim as optim
from lib.utils.util import check_path, empty_folder
from lib.utils.meter import AverageMeter
from torch.nn import DataParallel
from torch.backends import cudnn

__all__ = ['NetBase']


class NetBase(object):
    def __init__(self, nClass, nCam, model_client, use_flow, task_dir, raw_model_dir, is_image_dataset, recorder):
        self.nClass = nClass
        self.nCam = nCam
        self.recorder = recorder
        self.visual = self.recorder.visual
        self.logger = self.recorder.logger
        self._mode = 'Train'
        self.is_image_dataset = is_image_dataset
        self.task_dir = task_dir

        self.model = model_client(self.nClass, self.nCam, use_flow, self.is_image_dataset, raw_model_dir, self.logger)
        self.model_parallel = DataParallel(self.model).cuda()
        self.model_parallel.feature = DataParallel(self.model.feature).cuda()

        self.net_info = []
        self.const_options()
        self.init_options()
        self.loss_mean = AverageMeter(len(self.line_name))

        self.net_info.extend(self.model.net_info)
        self.optimizer = self.init_optimizer()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_step, gamma=self.gamma)
        self.idx = 0
        self.best_performance = 0.0

    def const_options(self):
        raise NotImplementedError

    def init_options(self):

        self.line_name = ['Identity Loss']

        raise NotImplementedError

    def init_optimizer(self):
        raise NotImplementedError

    def info(self, params_list):
        self.logger.info('------Output shape--------')
        for info_i in self.net_info:
            self.logger.info(info_i)
        if len(params_list) > 0:
            self.logger.info('----Parameter shape-------')
            for params_i in params_list:
                self.logger.info(params_i.size())

            self.logger.info('--------------------------')

    def display(self):
        info_str = 'Epoch: {} lr: {} Loss : '.format(self.idx, self.optimizer.param_groups[-1]['lr'])
        num = self.loss_mean.len
        for i in range(num):

            if i < num - 2:
                str_temp = ' + '
            elif i < num - 1:
                str_temp = ' = '
            else:
                str_temp = ''
            info_str += str(self.loss_mean[i].round(5)) + str_temp
        self.logger.info(info_str)

        self.visual.plot('Loss', 'lr', np.array([self.idx]), np.array([self.optimizer.param_groups[-1]['lr']]))

        for i_name, line_name in enumerate(self.line_name):
            self.visual.plot('Loss', line_name, np.array([self.idx]), self.loss_mean[i_name])
        self.loss_mean.reset()

    def forward(self, data):
        data = self.data_preprocess(data)
        model_output = self.model_parallel(data)
        return model_output

    def compute_loss(self, model_output, label_identify):
        raise NotImplementedError

    def forward_backward(self, args):
        self.scheduler.step(self.idx)
        data, label_identify = args
        model_output = self.forward(data)
        label_identify = self.data_preprocess(label_identify)
        loss_final = self.compute_loss(model_output, label_identify)
        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()

    def eval(self, args):
        data, label_identify = args
        data = self.data_preprocess(data)
        self.optimizer.zero_grad()
        fea = self.model_parallel.feature(data)
        return fea.detach()

    def sync(self, idx):
        self.idx = idx

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, flag):
        if flag == 'Train':
            self._mode = flag
            self.model_parallel.train()
            cudnn.benchmark = True
        elif flag == 'Test':
            self._mode = flag
            self.model_parallel.eval()
            cudnn.benchmark = True
        else:
            raise KeyError

    def save(self, rank1):
        if rank1 > self.best_performance:
            self.best_performance = rank1
            empty_folder(self.task_dir / 'output/model')
            torch.save(self.model_parallel.state_dict(), self.task_dir / 'output/model/model.pkl')
            self.logger.info('Model has been saved for index ' + str(self.idx))

    def load(self, model_name=None):
        if model_name is None:
            self.model_parallel.load_state_dict(torch.load(check_path(self.task_dir / 'output/model/model.pkl')))
            self.logger.info('Model restored from ' + str(self.task_dir / 'output/model/model.pkl'))
        else:
            self.model_parallel.load_state_dict(torch.load(check_path(self.task_dir / str('output/model/' + model_name + '.pkl'))))
            self.logger.info('Model restored from ' + str(self.task_dir / 'output/model/' + model_name + '.pkl'))

    def _data_group_prepocess(self, data):
        if not isinstance(data, (tuple, list)):
            if torch.is_tensor(data):
                return data.cuda()
            else:
                return torch.from_numpy(data).cuda()
        else:
            output = []
            for data_i in data:
                output.append(self._data_group_prepocess(data_i))
            return output

    def data_preprocess(self, *arg):
        output = []
        torch.set_grad_enabled(self._mode == 'Train')
        for data in arg:
            output.append(self._data_group_prepocess(data))

        output = tuple(output)
        if len(output) == 1:
            output = output[0]
        return output
