from lib.recorder.visual.torch_visdom import TorchVisual


class Visual(object):
    def __init__(self, platform, fpath, tag, device, logger):
        self._factory = {'Visdom': TorchVisual}
        self.device = device
        if platform not in self._factory:
            raise KeyError("Unknown platform:", platform)
        self.board = self._factory[platform](fpath, tag, self.device, logger)
        self.plot = self.board.plot
        self.finish = self.board.finish
        if hasattr(self.board, '_mode'):
            self.set_mode = self.board.set_mode