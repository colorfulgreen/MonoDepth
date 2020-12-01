import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.encoder = models.resnet18()
        self.conv1 = self.encoder.conv1

    def forward(self, x):
        '''返回所有层的特征，以便 decoder 中添加 identity connection'''
        out_conv1 = self.conv1(x)
        out_bn1 = self.encoder.bn1(out_conv1)
        out_relu1 = self.encoder.relu(out_bn1)

        out_layer1 = self.encoder.layer1(out_relu1)
        out_layer2 = self.encoder.layer2(out_layer1)
        out_layer3 = self.encoder.layer3(out_layer2)
        out_layer4 = self.encoder.layer4(out_layer3)
        return [out_relu1, out_layer1, out_layer2, out_layer3, out_layer4]


class ResNet18MultiImageInput(ResNet18):

    def __init__(self, n_images):
        super(ResNet18MultiImageInput, self).__init__()
        in_planes = 64
        self.conv1 = nn.Conv2d(n_images * 3, in_planes, kernel_size=7, stride=2, padding=1)
