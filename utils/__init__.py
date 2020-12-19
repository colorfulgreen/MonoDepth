import torch
import numpy as np

def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    '''Convert network's sigmoid output into depth predition'''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def imshow_tensors(tgt_img, projected_img, depth, batch_size):
    from PIL import Image
    from data.kitti_dataset import tensor_to_array
    import matplotlib.pyplot as plt
    batch_size = 3
    fig, ax = plt.subplots(batch_size, 3, figsize=(16, 8))
    for b in range(batch_size):
        array = tensor_to_array(tgt_img[b,:].clone().detach())
        im = Image.fromarray(array.astype(np.uint8))
        ax[b][0].imshow(im, cmap='gray')

        array = tensor_to_array(projected_img[b,:].clone().detach())
        im = Image.fromarray(array.astype(np.uint8))
        ax[b][1].imshow(im, cmap='gray')

        #abs_diff = torch.abs(tgt_img - projected_img)
        array = tensor_to_array(depth[b,:].clone().detach())
        im = Image.fromarray(array[:, :, 0].astype(np.uint8))
        ax[b][2].imshow(im, cmap='gray')

#import pdb; pdb.set_trace()
    plt.show()
