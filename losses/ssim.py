import torch
import torch.nn as nn

class SSIM(nn.Module):
    '''Layer to compute the SSIM loss between a pair of images'''

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        # mean intensity
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # use the standard deviation as an estimate of the signal contrast
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # overall similarity measure
        SSIM_n = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
