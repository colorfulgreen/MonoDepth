import numpy as np

def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    '''Convert network's sigmoid output into depth predition'''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def imshow_tensors(tensors):
    from PIL import Image
    from data.kitti_dataset import tensor_to_array
    import matplotlib.pyplot as plt
    n_imgs = len(tensors)
    for i, tensor in enumerate(tensors, start=1):
        array = tensor_to_array(tensor.detach())
        im = Image.fromarray(array.astype(np.uint8))
        plt.subplot(1, n_imgs, i)
        plt.imshow(im, cmap='gray')
    plt.show()


