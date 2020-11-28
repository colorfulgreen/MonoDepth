from path import Path

from torch.utils.data import DataLoader

from data.kitti_dataset import KITTIDataset

BASE_DIR = Path(__file__).realpath().dirname()
print(BASE_DIR)
# DATA_PATH = os.path.join(BASE_DIR, 'kitti_data')
DATA_PATH = '../monodepth2/kitti_data'
TRAIN_SPLITS_PATH = 'splits/eigen_zhou/train_files.txt'
VAL_SPLITS_PATH = 'splits/eigen_zhou/val_files.txt'

class Train(object):

    def train(self):
        with open(TRAIN_SPLITS_PATH) as f:
            data_splits = [i.strip() for i in f]
        train_set = KITTIDataset(DATA_PATH, data_splits)
        train_loader = DataLoader(train_set)
        for tgt_img, ref_imgs, K in train_loader:
            print(tgt_img)
            print('###########')
            break


if __name__ == '__main__':
    Train().train()
