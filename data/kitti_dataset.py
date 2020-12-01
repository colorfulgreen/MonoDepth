import os

import torch
import numpy as np
from imageio import imread
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset


# utils.image
def load_as_float(path):
    return imread(path).astype(np.float32)

# utils.transforms
def array_to_tensor(image):
    '''Convert numpy.ndarray (H x W x C) to a torch.FloatTensor (C x H x W)'''
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image / 255)

# utils.transforms
def resize(image, height, width):
    interp = Image.ANTIALIAS # TODO
    resize = transforms.Resize((height, width))
    resized_image = resize(Image.fromarray(image.astype(np.uint8)))
    return np.array(resized_image).astype(np.float32)


class KITTIDataset(Dataset):
    IMG_EXT = '.jpg'
    SIDE_MAP = {'l': 2, 'r': 3}
    WIDTH, HEIGHT = 640, 192

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

        # resize
        tgt_img = resize(tgt_img, self.HEIGHT, self.WIDTH)
        ref_imgs = [resize(img, self.HEIGHT, self.WIDTH) for img in ref_imgs]

        # array to tensor
        tgt_img = array_to_tensor(tgt_img)
        ref_imgs = [array_to_tensor(img) for img in ref_imgs]
        print('tgt_img<{}>, shape={}'.format(type(tgt_img), tgt_img.shape))

        # print(type(tgt_img), type(ref_imgs), type(self.K))
        return tgt_img, ref_imgs, self.K

    def load_image(self, folder, frame_index, side):
        fname = '{:010d}{}'.format(frame_index, self.IMG_EXT)
        fullpath = os.path.join(self.data_path,
                                folder,
                                'image_0{}/data'.format(self.SIDE_MAP[side]),
                                fname)
        print(fullpath)
        return load_as_float(fullpath)
