import torch.nn as nn

N_REFS = 2

class PoseDecoder(nn.Module):

    def __init__(self):
        super(PoseDecoder, self).__init__()

        self.conv3 = nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv1 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv0 = nn.Conv2d(256, 12, kernel_size=(1,1), stride=(1,1))
        self.relu = nn.ReLU()

    def forward(self, encode_features):
        last_features = encode_features[-1]

        # 将 n_planes 降到 6 * n_refs = 12, 然后对每个 plane 的特征取均值
        out_conv3 = self.relu(self.conv3(last_features))
        out_conv2 = self.relu(self.conv2(out_conv3))
        out_conv1 = self.relu(self.conv1(out_conv2))
        out_conv0 = self.conv0(out_conv1)
        pose = out_conv0.mean(3).mean(2)
        # TODO we follow [62] ... and scale the outputs by 0.01
        pose = 0.01 * out.view(-1, N_REFS, 1, 6)
        return pose
