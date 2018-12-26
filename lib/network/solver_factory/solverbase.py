from lib.network import NetManager
from lib.datamanager import DataManager
from lib.evaluation import Evaluator
from lib.utils.decoratorbox import vision_performance


class TaskSolverBase(object):
    def __init__(self, manager):
        self.manager = manager
        self.task_mode = self.manager.mode
        self.recorder = self.manager.recorder
        self.logger = self.manager.logger
        self.recorder.visual.set_mode(str(self.manager.split_id)+'_'+self.manager.dataset_name)

        self.const_options()
        self.init_options()
        self.logger.info('Instance num [{:2d}] Sample num [{:2d}] Depth [{:2d}]'.format(self.instance_num, self.sample_num, self.depth))
        self.Data = DataManager(name=self.manager.dataset_name,
                                root_dir=self.manager.device['root'],
                                rawfiles_dir=self.manager.device['rawfiles'],
                                split_id=self.manager.split_id,
                                use_flow=self.use_flow,
                                seed=self.manager.seed,
                                minframes=self.minframes,
                                num_workers=self.manager.device['num_workers'],
                                logger=self.logger)
        self.Data.set_train_generator(self.train_dataloder_type)
        self.Data.set_test_generator()
        self.evaluator = Evaluator(store_search_result=self.store_search_result,
                                   name=self.manager.dataset_name,
                                   task_dir=self.manager.task_dir,
                                   cuhk03_classic=self.manager.cuhk03_classic,
                                   logger=self.logger)
        self.perf_box = {}

    def const_options(self):
        # ------train-------
        self.display_step = 1
        self.instance_num = 8 if self.manager.device['name'] != 'pc' else 3
        self.sample_num = 4
        self.depth = 8 if self.manager.device['name'] != 'pc' else 2

        self.train_dataloder_type = 'All'
        self.minframes = {'train': self.depth, 'test': 1}

        # ------test--------
        self.test_batch_size = self.manager.device['test_batch_size']

    def init_options(self):
        # ------option------
        self.use_flow = False
        self.save_model = False
        self.reuse_model = False
        self.store_search_result = False
        self.net_client = None
        self.model_client = None

    def run(self):
        self.network = NetManager(nClass=self.Data.dataset.train_person_num,
                                  nCam=self.Data.dataset.train_cam_num,
                                  net_client=self.net_client,
                                  model_client=self.model_client,
                                  use_flow=self.use_flow,
                                  task_dir=self.manager.task_dir,
                                  raw_model_dir=self.manager.device['Model'],
                                  is_image_dataset=self.Data.dataset.is_image_dataset,
                                  recorder=self.recorder)
        if self.task_mode == 'Train':
            self.train_test()
        else:
            self.test()
        if self.store_search_result:
            self.evaluator.store_example()

    def train_test(self):
        network = self.network.create()
        train_flag, _ = self.manager.check_epoch(0)
        self.epoch = 0

        self.logger.info('Train  Begin')
        if self.reuse_model:
            network.load()

        network.mode = 'Train'
        self.Data.set_transform(network.model.get_transform())

        while train_flag:
            self.train(network)
            train_flag, test_flag = self.manager.check_epoch(self.epoch)

            if self.epoch % self.display_step == 0:
                network.display()

            if test_flag:
                cmc, mAP = self.do_eval(network, self.test_batch_size)

                network.mode = 'Train'
                self.Data.set_transform(network.model.get_transform())

                self.perf_box[str(self.epoch)] = {'cmc': cmc, 'mAP': mAP}
                if self.save_model:
                    network.save(cmc[0])

    def train(self, network):
        dataloader = self.Data.get_train(self.instance_num, self.sample_num, self.depth)
        for i_batch, batch_data in enumerate(dataloader):
            network.forward_backward(batch_data)
        self.epoch += 1
        network.sync(self.epoch)

    def test(self):
        network = self.network.create()
        network.load()
        self.do_eval(network, self.test_batch_size)

    @vision_performance
    def eval(self, network, batch_size):
        self.evaluator.reset(self.Data.dataset, distance_func=network.model.distance_func)
        self.evaluator.set_feature_buffer(network.model.fea_process_func)
        self.logger.info('Test Begin')
        network.mode = 'Test'
        self.Data.set_transform(network.model.get_transform())
        dataloader = self.Data.get_test(batch_size)
        for i_batch, batch_data in enumerate(dataloader):
            feature = network.eval(batch_data)
            self.evaluator.count(feature)

        result = self.evaluator.final_result()
        return result

    def do_eval(self, network, batch_size):
        result = self.eval(network, batch_size)
        cmc = result[0]
        mAP = result[1]
        cmc_rank = [float(cmc[0]), float(cmc[4]), float(cmc[9]), float(cmc[19])]
        return cmc_rank, mAP
