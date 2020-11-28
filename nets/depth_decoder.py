from torch.nn as nn
import torch.nn.functional as F


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1,1)),
        ELU(alpha=1.0, inplace=True)
    )

def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


class DepthDecoder(nn.Modules):

    def __init__(self):
        super(DepthDecoder, self).__init__()

        n_planes = [512, 256, 128, 64, 32, 16]
        self.upconv4 = upconv(n_planes[0], n_planes[1])
        self.iconv4 = upconv(n_planes[1], n_planes[1])
        self.upconv3 = upconv(n_planes[1], n_planes[2])
        self.iconv3 = upconv(n_planes[2], n_planes[2])
        self.upconv2 = upconv(n_planes[2], n_planes[3])
        self.iconv2 = upconv(n_planes[3], n_planes[3])
        self.upconv1 = upconv(n_planes[3], n_planes[4])
        self.iconv1 = upconv(n_planes[4], n_planes[4])
        self.upconv0 = upconv(n_planes[4], n_planes[5])
        self.iconv0 = upconv(n_planes[5], n_planes[5])

    def forward(self, encode_features):
        x = encoded_features[-1]
        out_upconv4 = self.upconv00(x)
        out_upconv4 = upsample(x)
        x = torch.cat(x)
