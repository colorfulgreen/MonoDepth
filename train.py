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
from losses import get_smooth_loss
from tensorboardX import SummaryWriter
from utils import disp_to_depth


BASE_DIR = Path(__file__).realpath().dirname()
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'
TRAIN_SPLITS_PATH = 'splits/eigen_zhou/train_files.txt'
#TRAIN_SPLITS_PATH = 'splits/eigen_zhou/debug_files.txt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Train(object):

    def __init__(self):
        self.W, self.H = 640, 192
        self.BATCH_SIZE = 12
        self.SCALES = [0, 1, 2, 3]
        self.SCHEDULER_STEP_SIZE = 15
        self.N_EPOCHS = 20
        self.MIN_DEPTH = 0.1
        self.MAX_DEPTH = 100

        with open(TRAIN_SPLITS_PATH) as f:
            data_splits = [i.strip() for i in f]
        train_set = KITTIDataset(DATA_PATH, data_splits, self.W, self.H, device, [-1, 1])
        self.train_loader = DataLoader(train_set, self.BATCH_SIZE, drop_last=True)

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
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.SCHEDULER_STEP_SIZE, gamma=0.1)

        self.n_iter = 0
        self.tb_writer = SummaryWriter()

    def train(self):
        for epoch in range(self.N_EPOCHS):
            print('\n\n>>>>>>>>>>> EPOCH {} >>>>>>>>>>>>\n\n'.format(epoch))
            self.run_epoch()

    def run_epoch(self):
        self.lr_scheduler.step()
        for i, (tgt_img, ref_imgs, intrinsics) in enumerate(self.train_loader):
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
            self.tb_writer.add_scalar('loss', loss.item(), self.n_iter)

            if i % 10 == 0:
                print('ITER {} LOSS {}'.format(i, loss))

        self.save_model()

    def compute_losses(self, tgt_img, ref_imgs, intrinsics, disparities, poses):
        total_loss = 0
        for scale in self.SCALES:
            disp = disparities[scale]

            reprojection_loss = self.photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, disp, poses)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            b, _, h, w = disp.shape
            scaled_tgt_img = F.interpolate(tgt_img, (h, w), mode='area')
            smooth_loss = 1e-3 * get_smooth_loss(norm_disp, scaled_tgt_img)

            total_loss += reprojection_loss + smooth_loss / (2**scale) # TODO 2**scale

            self.tb_writer.add_scalar('smooth_loss_scale{}'.format(scale), smooth_loss/(2**scale), self.n_iter)

        return total_loss / len(self.SCALES)

    def photometric_reconstruction_loss(self, tgt_img, ref_imgs, intrinsics, disp, poses):
        self.tb_writer.add_scalar('disp_mean', disp.mean(), self.n_iter)
        if disp.shape != tgt_img.shape: # NOTE upsample the lower resolution depth map
            disp = F.interpolate(disp, [self.H, self.W], mode='bilinear', align_corners=False)
        _, depth = disp_to_depth(disp, self.MIN_DEPTH, self.MAX_DEPTH)

        reprojection_losses = []
        cam_coords = self.backproject_depth(depth, intrinsics.inverse())
        for i, ref_img in enumerate(ref_imgs):
            pose = poses[:, i]     # [B, 6]
            T = pose_vec2mat(pose)
            pixel_coords = self.project_3d(cam_coords, intrinsics, T)
            projected_img = F.grid_sample(ref_img, pixel_coords, padding_mode='zeros', align_corners=True)
            reprojection_losses.append(self.appearance_similarity(tgt_img, projected_img))
        reprojection_losses = torch.cat(reprojection_losses, dim=1)

        # compute reprojection error of the original for auto-masking
        identity_reprojection_losses = []
        for i, ref_img in enumerate(ref_imgs):
            identity_reprojection_losses.append(self.appearance_similarity(tgt_img, ref_img))
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, dim=1)
        identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape).cuda() * 0.00001 # add random numbe r to break ties?

        # auto-masking that fillters out pixels which do not change appearance from one frame to the next
        combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
        min_reprojection_losses, min_idxs = torch.min(combined, dim=1)
        automask = (min_idxs > identity_reprojection_losses.shape[1] - 1).float()

        # loss = (automask * min_reprojection_losses).mean()  # TODO automask
        loss = min_reprojection_losses.mean()

        self.tb_writer.add_scalar('reprojection_loss', reprojection_losses.mean(), self.n_iter)
        self.tb_writer.add_scalar('min_reprojection_loss', min_reprojection_losses.mean(), self.n_iter)

        if False: # and depth.mean() < 0.1001: # and loss < 0.05:
            print('REPROJECTION LOSS:', loss)
            print('DEPTH MEAN:', depth.mean())
            print('POSE MEAN:', pose.mean())
            from utils import imshow_tensors
            imshow_tensors(tgt_img, projected_img, depth, self.BATCH_SIZE)
            import pdb; pdb.set_trace()

        return loss

    def appearance_similarity(self, tgt_img, projected_img):
        abs_diff = torch.abs(tgt_img - projected_img)
        l1_loss = abs_diff.mean(1, True) # [12, 1, 192, 640]
        ssim_loss = self.ssim(projected_img, tgt_img).mean(1, True)
        similarity = 0.85 * ssim_loss + 0.15 * l1_loss
        self.tb_writer.add_scalar('l1_loss', l1_loss.mean(), self.n_iter)
        self.tb_writer.add_scalar('ssim_loss', ssim_loss.mean(), self.n_iter)
        return similarity   # [12, 1, 192, 640]

    def save_model(self):
        folder = os.path.join(BASE_DIR, 'runs', 'models')
        torch.save(self.depth_encoder, os.path.join(folder, 'depth_encoder.pth'))
        torch.save(self.depth_decoder, os.path.join(folder, 'depth_decoder.pth'))
        torch.save(self.pose_encoder, os.path.join(folder, 'pose_encoder.pth'))
        torch.save(self.pose_encoder, os.path.join(folder, 'pose_decoder.pth'))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    Train().train()
