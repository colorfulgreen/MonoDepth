import os

import torch
import numpy as np
from imageio import imread

from torch.utils.data import Dataset


# utils.image
def load_as_float(path):
    return imread(path).astype(np.float32)


class KITTIDataset(Dataset):
    IMG_EXT = '.jpg'
    SIDE_MAP = {'l': 2, 'r': 3}

    def __init__(self, data_path, data_splits):
        self.data_path = data_path
        self.data_splits = data_splits
        # TODO intrinsics
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def __len__(self):
        return len(self.data_splits)

    def __getitem__(self, index):
        # TODO batch_size = 12, n_scales = 4
        folder, frame_index, side = self.data_splits[index].split()
        frame_index = int(frame_index)
        tgt_img = self.load_image(folder, frame_index, side)
        ref_imgs = [self.load_image(folder, frame_index - 1, side),
                    self.load_image(folder, frame_index - 1, side)]
        print(type(tgt_img), type(ref_imgs), type(self.K))
        return tgt_img, ref_imgs, self.K

    def load_image(self, folder, frame_index, side):
        fname = '{:010d}{}'.format(frame_index, self.IMG_EXT)
        fullpath = os.path.join(self.data_path,
                                folder,
                                'image_0{}/data'.format(self.SIDE_MAP[side]),
                                fname)
        array = load_as_float(fullpath)
        return torch.from_numpy(array) # TODO / 255
