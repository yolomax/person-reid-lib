class NetManager(object):
    def __init__(self, nClass, nCam, net_client, model_client, use_flow, task_dir, raw_model_dir, is_image_dataset, recorder):
        self.nClass = nClass
        self.nCam = nCam
        self._net_client = net_client
        self._model_client = model_client
        self.task_dir = task_dir
        self.raw_model_dir = raw_model_dir
        self.use_flow = use_flow
        self.is_image_dataset = is_image_dataset
        self.recorder = recorder

    def create(self):
        return self._net_client(self.nClass, self.nCam, self._model_client, self.use_flow, self.task_dir, self.raw_model_dir, self.is_image_dataset, self.recorder)