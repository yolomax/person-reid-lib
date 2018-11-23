import numpy as np
import cv2


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def cal_for_frames(frames, resize_hw, opfunc=compute_TVL1):

    images = []
    prev = cv2.imread(frames[0])
    if resize_hw is not None:
        prev = cv2.resize(prev, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        if resize_hw is not None:
            curr = cv2.resize(curr, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = opfunc(prev_gray, curr_gray)
        images.append(np.concatenate((curr[..., ::-1],tmp_flow), axis=-1).astype('uint8'))
        prev_gray = curr_gray

    return images


class OpticalFlowManager(object):
    def __init__(self, img_dir_list, track_info, resize_hw, flow_func='TVL1'):
        self._of_factory = {'TVL1': compute_TVL1}
        self.img_dir_list = img_dir_list
        self.len = len(self.img_dir_list)
        self.track_info = track_info
        self.resize_hw = resize_hw
        assert self.track_info[:, -1].sum() == self.len
        assert self.track_info[-1, 3] == self.len
        assert self.track_info[0, 2] == 0
        raw_begin_list = self.track_info[:, 2].copy()
        sorted_begin_list = np.sort(raw_begin_list.copy())
        assert (raw_begin_list == sorted_begin_list).all()

        if flow_func in self._of_factory:
            self._func = self._of_factory[flow_func]
        else:
            raise KeyError

        self.begin_idx = self.track_info[0, 2]
        self.end_idx = self.track_info[0, 3]
        self.track_idx = 0
        self.images_and_of = cal_for_frames(self.img_dir_list[self.begin_idx:self.end_idx], self.resize_hw, self._func)

    def __call__(self, idx):
        if not self.begin_idx <= idx < self.end_idx:
            self.track_idx += 1
            self.begin_idx = self.track_info[self.track_idx, 2]
            self.end_idx = self.track_info[self.track_idx, 3]
            self.images_and_of = cal_for_frames(self.img_dir_list[self.begin_idx:self.end_idx], self.resize_hw, self._func)
        return self.images_and_of[idx - self.begin_idx]


