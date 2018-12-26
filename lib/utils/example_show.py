import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image


class SearchShow(object):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.false_example = []
        self.right_example = []
        self.read()

    def read(self):
        with h5py.File(self.file_dir,'r') as f:
            num = f['num'][...]
            for i_epoch in range(num):
                false_box = []
                right_box = []
                false_box.append(f['false_img_' + str(i_epoch)][...])
                false_box.append(f['false_dst_' + str(i_epoch)][...])
                false_box.append(f['false_id_' + str(i_epoch)][...])
                right_box.append(f['right_img_' + str(i_epoch)][...])
                right_box.append(f['right_dst_' + str(i_epoch)][...])
                right_box.append(f['right_id_' + str(i_epoch)][...])
                self.false_example.append(false_box)
                self.right_example.append(right_box)

    def __call__(self, example, epoch_id,person_id):
        img_num = example[0][0].shape[1]
        fig = plt.figure(1)
        for i_image in range(img_num):
            img_dir = str(example[epoch_id][0][person_id, i_image])[1:].strip('\'')
            # img_dir = img_dir.replace('/server_root', '/pc_root')    # replace the server image dir by local PC image dir
            img = Image.open(img_dir).convert('RGB')
            ax = fig.add_subplot(2,(img_num+1)//2, i_image+1)
            ax.imshow(img)
            if i_image == 0:
                id_info = example[epoch_id][2][person_id, i_image]
                ax.set_title('Probe #' + str(id_info))
            else:
                dst_info = example[epoch_id][1][person_id,i_image-1]
                id_info = example[epoch_id][2][person_id, i_image]
                ax.set_title(str(id_info) + '#\n' + str(round(dst_info, 3)))
        plt.show()

