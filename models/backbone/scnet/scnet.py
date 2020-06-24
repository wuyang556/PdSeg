import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
import paddle
import paddle.fluid as fluid
# from .utils import ReLU
from PaddleVision.models.utils import ReLU


__all__ = ['SCNet', 'scnet50', 'scnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return fluid.dygraph.Conv2D(in_planes, out_planes, filter_size=3, stride=stride,
                                padding=1, groups=groups, bias_attr=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return fluid.dygraph.Conv2D(in_planes, out_planes, filter_size=1, stride=stride, bias_attr=False)


class SCConv(fluid.dygraph.Layer):
    def __init__(self, planes, stride, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = fluid.dygraph.container.Sequential(
                    fluid.dygraph.Pool2D(pool_type="avg", pool_size=pooling_r, pool_stride=pooling_r),
                    conv3x3(planes, planes), 
                    fluid.dygraph.BatchNorm(planes),
                    )
        self.k3 = fluid.dygraph.container.Sequential(
                    conv3x3(planes, planes), 
                    fluid.dygraph.BatchNorm(planes),
                    )
        self.k4 = fluid.dygraph.container.Sequential(
                    conv3x3(planes, planes, stride), 
                    fluid.dygraph.BatchNorm(planes),
                    ReLU(),
                    )

    def forward(self, x):
        identity = x
        # out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = fluid.layers.sigmoid(identity + fluid.layers.resize_bilinear(self.k2(x), out_shape=identity.shape[2:]))
        out = fluid.layers.matmul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(fluid.dygraph.Layer):
    expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = conv1x1(inplanes, planes)
        self.bn1_a = fluid.dygraph.BatchNorm(planes)

        self.k1 = fluid.dygraph.container.Sequential(
                    conv3x3(planes, planes, stride), 
                    fluid.dygraph.BatchNorm(planes),
                    ReLU(),
                    )

        self.conv1_b = conv1x1(inplanes, planes)
        self.bn1_b = fluid.dygraph.BatchNorm(planes)
        
        self.scconv = SCConv(planes, stride, self.pooling_r)

        self.conv3 = conv1x1(planes * 2, planes * 2 * self.expansion)
        self.bn3 = fluid.dygraph.BatchNorm(planes * 2 * self.expansion)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(fluid.layers.concat([out_a, out_b], axis=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SCNet(fluid.dygraph.Layer):
    def __init__(self, block, layers, num_classes=1000):
        super(SCNet, self).__init__()
        self.inplanes = 64
        self.conv1 = fluid.dygraph.Conv2D(3, 64, filter_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = fluid.dygraph.BatchNorm(self.inplanes)
        self.relu = ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = fluid.dygraph.Pool2D(pool_type="max", pool_size=3, pool_stride=2, pool_padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = fluid.dygraph.Pool2D(pool_type="avg", global_pooling=True, pool_size=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = fluid.dygraph.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fluid.dygraph.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                fluid.dygraph.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x


def scnet50(pretrained=False, root="~/work/PdSeg/models", **kwargs):
    """Constructs a SCNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model_path = "E:\Code\Python\PaddleSeg\SCNet\scnet50-dc6a7e87.pth"
        model_path = os.path.join(os.path.expanduser(root), "backbone/scnet", "scnet50")
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
        # model.load_state_dict(model_zoo.load_url(model_urls['scnet50']))
    return model


def scnet101(pretrained=False, root="~/work/PdSeg/models", **kwargs):
    """Constructs a SCNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model_path = "E:\Code\Python\PaddleSeg\SCNet\scnet101-44c5b751.pth"
        model_path = os.path.join(os.path.expanduser(root), "backbone/scnet", "scnet101")
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
        # model.load_state_dict(model_zoo.load_url(model_urls['scnet101']))
    return model


if __name__ == '__main__':
    # images = torch.rand(4, 3, 224, 224)
    # model = scnet101(pretrained=True)
    # model = model
    # print(model(images).size())
    with fluid.dygraph.guard():
        import numpy as np
        in_np = np.ones(shape=[4, 3, 224, 224], dtype="float32")
        in_var = fluid.dygraph.to_variable(in_np)
        model = scnet101(pretrained=False)
        out = model(in_var)
        print(out.shape)
