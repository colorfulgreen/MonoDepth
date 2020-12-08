import os
from path import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from nets.resnet import ResNet18, ResNet18MultiImageInput
from nets.depth_decoder import DepthDecoder
from nets.pose_decoder import PoseDecoder
from losses.photometric_reconstruction_loss import BackprojectDepth, Project3D, pose_vec2mat
from losses.ssim import SSIM
from tensorboardX import SummaryWriter


BASE_DIR = Path(__file__).realpath().dirname()
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'
TRAIN_SPLITS_PATH = 'splits/eigen_zhou/train_files.txt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Train(object):

    def __init__(self):
        self.W, self.H = 640, 192
        self.BATCH_SIZE = 12
        self.SCALES = [0, 1, 2, 3]
        self.backproject_depth = BackprojectDepth(self.W, self.H, device, self.BATCH_SIZE)
        self.ssim = SSIM()
        self.ssim.to(device)

        self.project_3d = Project3D(self.W, self.H, self.BATCH_SIZE)

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

        self.n_iter = 0
        self.tb_writer = SummaryWriter()

    def train(self):
        with open(TRAIN_SPLITS_PATH) as f:
            data_splits = [i.strip() for i in f]
        train_set = KITTIDataset(DATA_PATH, data_splits, self.W, self.H, device, [-1, 1])
        train_loader = DataLoader(train_set, self.BATCH_SIZE)

        for i, (tgt_img, ref_imgs, intrinsics) in enumerate(train_loader):
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            disp_features = self.depth_encoder(tgt_img)
            disparities = self.depth_decoder(disp_features)
            pose_features = self.pose_encoder(torch.cat([tgt_img] + ref_imgs, 1))
            poses = self.pose_decoder(pose_features)

            loss = self.compute_losses(tgt_img, ref_imgs, intrinsics, disparities, poses)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.n_iter += 1
            self.tb_writer.add_scalar('photometric_error', loss.item(), self.n_iter)

            if i % 100 == 0:
                print(i)

        self.save_model()

    def compute_losses(self, tgt_img, ref_imgs, intrinsics, disparities, poses):
        total_loss = 0
        for scale in self.SCALES:
            disp = disparities[scale]
            reprojection_loss = self.photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, disp, poses)
            total_loss += reprojection_loss
        return total_loss / len(self.SCALES)


    def photometric_reconstruction_loss(self, tgt_img, ref_imgs, intrinsics, disp, poses):
        if disp.shape != tgt_img.shape:
            disp = F.interpolate(disp, [self.H, self.W], mode='bilinear', align_corners=False)
        losses = []
        depth = 1 / disp # TODO depth
        cam_coords = self.backproject_depth(depth, intrinsics.inverse())

        for i, ref_img in enumerate(ref_imgs):
            pose = poses[:, i]     # [B, 6]
            T = pose_vec2mat(pose)
            pixel_coords = self.project_3d(cam_coords, intrinsics, T)
            projected_img = F.grid_sample(tgt_img, pixel_coords, padding_mode='zeros', align_corners=True)
            abs_diff = torch.abs(tgt_img - projected_img)
            l1_loss = abs_diff.mean(1, True) # [12, 1, 192, 640]
            ssim_loss = self.ssim(projected_img, tgt_img)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
            losses.append(reprojection_loss)
        min_loss, _ = torch.min(torch.cat(losses, dim=1), dim=1)
        # import pdb; pdb.set_trace()
        return min_loss.mean()

    def save_model(self):
        folder = os.path.join(BASE_DIR, 'runs', 'models')
        torch.save(self.depth_encoder, os.path.join(folder, 'depth_encoder.pth'))
        torch.save(self.depth_decoder, os.path.join(folder, 'depth_decoder.pth'))
        torch.save(self.pose_encoder, os.path.join(folder, 'pose_encoder.pth'))
        torch.save(self.pose_encoder, os.path.join(folder, 'pose_decoder.pth'))


if __name__ == '__main__':
    Train().train()
