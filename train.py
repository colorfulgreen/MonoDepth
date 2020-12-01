from path import Path

import torch
from torch.utils.data import DataLoader

from data.kitti_dataset import KITTIDataset
from nets.resnet import ResNet18, ResNet18MultiImageInput
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

        depth_encoder = ResNet18()
        depth_decoder = DepthDecoder()
        depth_encoder.to(device)
        depth_decoder.to(device)

        pose_encoder = ResNet18MultiImageInput(n_images=3)
        pose_decoder = PoseDecoder()
        pose_encoder.to(device)
        pose_decoder.to(device)

        for tgt_img, ref_imgs, K in train_loader:
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            disp_features = depth_encoder(tgt_img)
            disparities = depth_decoder(disp_features)

            # import pdb; pdb.set_trace()
            pose_features = pose_encoder(torch.cat([tgt_img] + ref_imgs, 1))
            poses = pose_decoder(pose_features)

        loss = photometric_reconstruction_loss(tgt_img, ref_imgs, K, disparities, poses)


if __name__ == '__main__':
    Train().train()
