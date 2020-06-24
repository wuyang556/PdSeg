import paddle.fluid as fluid

from .base import BaseNet
from .fcn import FCNHead
from .nn import PyramidPooling, ReLU, Dropout2d


class PSPNet(BaseNet):
    def __init__(self, args, nclass, backbone, backbone_style, aux=True, se_loss=False, norm_layer=fluid.dygraph.BatchNorm, **kwargs):
        super(PSPNet, self).__init__(args, nclass, backbone, backbone_style, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPNetHead(512*len(args.dilations), nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.shape
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = fluid.layers.resize_bilinear(input=x, out_shape=(h, w))
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = fluid.layers.resize_bilinear(input=auxout, out_shape=(h, w))
            outputs.append(auxout)
        return tuple(outputs)


class PSPNetHead(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = fluid.dygraph.container.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                                        fluid.dygraph.Conv2D(in_channels * 2,
                                                                             inter_channels, 3,
                                                                             padding=1, bias_attr=False),
                                                        norm_layer(inter_channels),
                                                        ReLU(),
                                                        Dropout2d(0.1),
                                                        fluid.dygraph.Conv2D(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

