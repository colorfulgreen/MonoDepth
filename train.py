from path import Path

import torch
from torch.utils.data import DataLoader

from data.kitti_dataset import KITTIDataset
from nets.resnet import ResNet18
from nets.depth_decoder import DepthDecoder
from nets.pose_decoder import PoseDecoder


BASE_DIR = Path(__file__).realpath().dirname()
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'
TRAIN_SPLITS_PATH = 'splits/eigen_zhou/train_files.txt'
VAL_SPLITS_PATH = 'splits/eigen_zhou/val_files.txt'


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Train(object):

    def train(self):
        with open(TRAIN_SPLITS_PATH) as f:
            data_splits = [i.strip() for i in f]
        train_set = KITTIDataset(DATA_PATH, data_splits)
        train_loader = DataLoader(train_set)
        encoder = ResNet18()
        depth_decoder = DepthDecoder()
        pose_decoder = PoseDecoder()

        for tgt_img, ref_imgs, K in train_loader:
            import pdb; pdb.set_trace()
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            disp_features = encoder(tgt_img)
            disparities = (disp_features)

            pose_features = ResNet18(torch.cat([tgt_img] + ref_imgs, 1))
            poses = PoseDecoder(pose_features)

        loss = photometric_reconstruction_loss(tgt_img, ref_imgs, K, disparities, poses)


if __name__ == '__main__':
    Train().train()
