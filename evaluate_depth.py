import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import DATA_PATH, DEPTH_ENCODER_PATH, DEPTH_DECODER_PATH, W, H
from data.kitti_dataset import KITTIDataset, resize
from nets.resnet import ResNet18
from nets.depth_decoder import DepthDecoder
from metrics.depth import depth_error_and_accuracy_metric
from utils.color import trans_colormapped_depth_image
from utils import disp_to_depth, imshow_tensors

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TEST_SPLITS_PATH = 'splits/eigen/test_files.txt'
GT_PATH = 'splits/eigen/gt_depths.npz'

MIN_DEPTH = 1E-3
MAX_DEPTH = 80

def imshow_results(tgt_img, pred_disp, gt_depth):
    from PIL import Image
    from data.kitti_dataset import tensor_to_array
    import pdb; pdb.set_trace()
    _, pred_depth = disp_to_depth(pred_disp)
    im_pred = trans_colormapped_depth_image(pred_depth)
#im_gt = trans_colormapped_depth_image(gt_depth)
#im_pred = Image.fromarray(pred_depth.astype(np.uint8))
    plt.subplot(1,2,1)
    plt.imshow(Image.fromarray(tensor_to_array(tgt_img).astype(np.uint8)))
    plt.subplot(1,2,2)
    plt.imshow(im_pred, cmap='gray')
    plt.show()


@torch.no_grad()
def evaluate_depth():

    with open(TEST_SPLITS_PATH) as f:
        splits = [i.strip() for i in f]
    dataset = KITTIDataset(DATA_PATH, splits, W, H, device, [])
    dataloader = DataLoader(dataset)
    gt_depths = np.load(GT_PATH, allow_pickle=True, encoding='latin1')['data']

    depth_encoder = torch.load(DEPTH_ENCODER_PATH)
    depth_decoder = torch.load(DEPTH_DECODER_PATH)
    depth_encoder.to(device)
    depth_decoder.to(device)

    pred_disps = []
    for i, (tgt_img, ref_imgs, intrinsics) in enumerate(dataloader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]

        disp_features = depth_encoder(tgt_img)
        pred_disp = depth_decoder(disp_features)
        import pdb; pdb.set_trace()
        pred_disps.append(pred_disp[-1][0, 0, :].cpu().numpy()) # TODO scales

        if True:
            imshow_results(tgt_img, pred_disps[-1], gt_depths[i])

        if i % 100 == 0: print(i)

    errors = []
    for i, pred_disp in enumerate(pred_disps):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = resize(pred_disp, gt_height, gt_width)
        _, pred_depth = disp_to_depth(pred_disp)

        # TODO
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(depth_error_and_accuracy_metric(gt_depth, pred_depth))

    print(np.array(errors).mean(0))


if __name__ == '__main__' :
    evaluate_depth()
