import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Modules):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.encoder = models.resnet18()

    def forward(self, x):
        '''返回所有层的特征，以便 decoder 中添加 identity connection'''
        out_conv1 = self.encoder.conv1(x)
        out_bn1 = self.encoder.bn1(out_conv1)
        out_relu1 = self.encoder.relu(out_bn1)

        out_layer1 = self.encoder.layer1(out_relu1)
        out_layer2 = self.encoder.layer2(out_layer1)
        out_layer3 = self.encoder.layer2(out_layer2)
        out_layer4 = self.encoder.layer2(out_layer3)
        return [out_relu1, out_layer1, out_layer2, out_layer3, out_layer4]
