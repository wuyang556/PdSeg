import paddle
import paddle.fluid as fluid

from .fcn import FCNHead
from .base import BaseNet
from .nn import ReLU, Dropout2d, AdaptiveAvgPool2d


class DeepLabV3(BaseNet):
    def __init__(self, args, nclass, backbone, backbone_style, aux=True, se_loss=False, norm_layer=fluid.dygraph.BatchNorm, **kwargs):
        super(DeepLabV3, self).__init__(args, nclass, backbone, backbone_style, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.shape
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = fluid.layers.resize_bilinear(x, out_shape=(h, w))
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = fluid.layers.resize_bilinear(auxout, out_shape=(h, w))
            outputs.append(auxout)

        return tuple(outputs)


class DeepLabV3Head(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.block = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(inter_channels, inter_channels, 3, padding=1, bias_attr=False),
            norm_layer(inter_channels),
            ReLU(),
            Dropout2d(0.1),
            fluid.dygraph.Conv2D(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = fluid.dygraph.container.Sequential(
        fluid.dygraph.Conv2D(in_channels, out_channels, 3, padding=atrous_rate,
                             dilation=atrous_rate, bias_attr=False),
        norm_layer(out_channels),
        ReLU())
    return block


class AsppPooling(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = fluid.dygraph.container.Sequential(
            AdaptiveAvgPool2d(1),
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())

    def forward(self, x):
        _, _, h, w = x.shape
        pool = self.gap(x)

        return fluid.layers.resize_bilinear(pool, out_shape=(h, w))


class ASPP_Module(fluid.dygraph.Layer):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(5*out_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU(),
            Dropout2d(0.5))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = fluid.layers.concat([feat0, feat1, feat2, feat3, feat4], 1)

        return self.project(y)

