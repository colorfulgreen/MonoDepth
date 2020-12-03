from path import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from nets.resnet import ResNet18, ResNet18MultiImageInput
from nets.depth_decoder import DepthDecoder
from nets.pose_decoder import PoseDecoder
from losses.photometric_reconstruction_loss import BackprojectDepth, Project3D, pose_vec2mat


BASE_DIR = Path(__file__).realpath().dirname()
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'
TRAIN_SPLITS_PATH = 'splits/eigen_zhou/train_files.txt'
VAL_SPLITS_PATH = 'splits/eigen_zhou/val_files.txt'


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Train(object):

    def __init__(self):
        self.W, self.H = 640, 192
        self.backproject_depth = BackprojectDepth(self.W, self.H, device)
        self.project_3d = Project3D(self.W, self.H)

        self.depth_encoder = ResNet18()
        self.depth_decoder = DepthDecoder()
        self.pose_encoder = ResNet18MultiImageInput(n_images=3)
        self.pose_decoder = PoseDecoder()

        self.depth_encoder.to(device)
        self.depth_decoder.to(device)
        self.pose_encoder.to(device)
        self.pose_decoder.to(device)

        optim_params = [
            {'params': self.depth_encoder.parameters()},
            {'params': self.depth_decoder.parameters()},
            {'params': self.pose_encoder.parameters()},
            {'params': self.pose_decoder.parameters()}
        ]
        self.optimizer = torch.optim.Adam(optim_params, lr=1e-4)

    def train(self):
        with open(TRAIN_SPLITS_PATH) as f:
            data_splits = [i.strip() for i in f]
        train_set = KITTIDataset(DATA_PATH, data_splits, self.W, self.H, device)
        train_loader = DataLoader(train_set)


        for tgt_img, ref_imgs, intrinsics in train_loader:
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            disp_features = self.depth_encoder(tgt_img)
            disparities = self.depth_decoder(disp_features)
            pose_features = self.pose_encoder(torch.cat([tgt_img] + ref_imgs, 1))
            poses = self.pose_decoder(pose_features)

            loss = self.photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, disparities, poses)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('LOSS:', loss)

    def photometric_reconstruction_loss(self, tgt_img, ref_imgs, intrinsics, disparities, poses):
        losses = []
        depth = 1 / disparities[-1] # TODO depth & scales
        cam_coords = self.backproject_depth(depth, intrinsics.inverse())

        for i, ref_img in enumerate(ref_imgs):
            pose = poses[:, i]     # [B, 6]
            T = pose_vec2mat(pose)
            pixel_coords = self.project_3d(cam_coords, intrinsics, T)
            projected_img = F.grid_sample(tgt_img, pixel_coords, padding_mode='zeros', align_corners=True)
            abs_diff = torch.abs(tgt_img - projected_img)
            l1_loss = abs_diff.mean(1, True)
            losses.append(l1_loss)
        min_loss, _ = torch.min(torch.cat(losses, dim=1), dim=1)
        # import pdb; pdb.set_trace()
        return min_loss.mean()


if __name__ == '__main__':
    Train().train()
