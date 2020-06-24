# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/5/21

import paddle.fluid as fluid
# from ..nn import ReLU
from PdSeg.models.nn import ReLU


class SeparableConv2d(fluid.dygraph.Layer):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=fluid.dygraph.BatchNorm):
        super(SeparableConv2d, self).__init__()

        self.conv1 = fluid.dygraph.Conv2D(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias_attr=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = fluid.dygraph.Conv2D(inplanes, planes, 1, 1, 0, 1, 1, bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JUM(fluid.dygraph.Layer):
    def __init__(self, in_channels, width, dilation, norm_layer, up_kwargs):
        super(JUM, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv_l = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())
        self.conv_h = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())

        norm_layer = lambda n_channels: fluid.dygraph.GroupNorm(groups=32, channels=n_channels)
        self.dilation1 = fluid.dygraph.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=dilation, dilation=dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       ReLU())
        self.dilation2 = fluid.dygraph.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2*dilation, dilation=2*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       ReLU())
        self.dilation3 = fluid.dygraph.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=4*dilation, dilation=4*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       ReLU())

    def forward(self, x_l, x_h):
        print(self.conv_l(x_l).shape)
        print(self.conv_h(x_h).shape)
        feats = [self.conv_l(x_l), self.conv_h(x_h)]
        _, _, h, w = feats[-1].shape
        feats[-2] = fluid.layers.resize_bilinear(feats[-2], out_shape=(h, w))
        feat = fluid.layers.concat(feats, axis=1)
        feat = fluid.layers.concat([feats[-2], self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], axis=1)
        print(feat.shape)
        return feat


class JPU_X(fluid.dygraph.Layer):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU_X, self).__init__()
        self.jum_1 = JUM(in_channels[:2], width//2, 1, norm_layer, up_kwargs)
        self.jum_2 = JUM(in_channels[1:], width, 2, norm_layer, up_kwargs)

    def forward(self, *inputs):
        feat = self.jum_1(inputs[2], inputs[1])
        feat = self.jum_2(inputs[3], feat)

        return inputs[0], inputs[1], inputs[2], feat


if __name__ == '__main__':
    import numpy as np
    inputs = np.ones((4, 256, 128, 128), dtype="float32")
    inputs0 = np.ones((4, 512, 64, 64), dtype="float32")
    inputs1 = np.ones((4, 1024, 32, 32), dtype="float32")
    inputs2 = np.ones((4, 2048, 16, 16), dtype="float32")
    with fluid.dygraph.guard():
        inputs = fluid.dygraph.to_variable(inputs)
        inputs0 = fluid.dygraph.to_variable(inputs0)
        inputs1 = fluid.dygraph.to_variable(inputs1)
        inputs2 = fluid.dygraph.to_variable(inputs2)
        model = JPU_X(in_channels=(512, 1024, 2048), norm_layer=fluid.dygraph.BatchNorm, up_kwargs={"mode": "bilinear", "align_corners": True})
        out = model(*[inputs, inputs0, inputs1, inputs2])
        print(out[-1].shape)
