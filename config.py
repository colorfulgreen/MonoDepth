import os
from path import Path

BASE_DIR = Path(__file__).realpath().dirname()
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'

GT_PATH = 'splits/eigen/'

_MODEL_DIR = os.path.join(BASE_DIR, 'runs', 'models')
DEPTH_ENCODER_PATH = os.path.join(_MODEL_DIR, 'depth_encoder.pth')
DEPTH_DECODER_PATH = os.path.join(_MODEL_DIR, 'depth_decoder.pth')

W, H = 640, 192
