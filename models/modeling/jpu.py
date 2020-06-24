# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24
import paddle
import paddle.fluid as fluid
from PdSeg.models.nn import ReLU, Dropout2d


class SeparableConv2D(fluid.dygraph.Layer):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2D, self).__init__()

        self.conv1 = fluid.dygraph.Conv2D(
            num_channels=inplanes,
            num_filters=inplanes,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=inplanes,
            bias_attr=bias)
        self.bn = fluid.dygraph.BatchNorm(inplanes)
        self.pointwise = fluid.dygraph.Conv2D(
            num_channels=inplanes,
            num_filters=planes,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(fluid.dygraph.Layer):
    """
    Joint Parallel Unit.
    """
    def __init__(self, in_channels, width=512, dilations=(1, 2, 4, 8), norm_layer=None):
        super(JPU, self).__init__()

        self.conv5 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())
        self.conv4 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())
        self.conv3 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-3], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())

        # self.dilation1 = fluid.dygraph.container.Sequential(
        #     SeparableConv2D(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
        #     norm_layer(width),
        #     ReLU())
        # self.dilation2 = fluid.dygraph.container.Sequential(
        #     SeparableConv2D(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
        #     norm_layer(width),
        #     ReLU())
        # self.dilation3 = fluid.dygraph.container.Sequential(
        #     SeparableConv2D(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
        #     norm_layer(width),
        #     ReLU())
        # self.dilation4 = fluid.dygraph.container.Sequential(
        #     SeparableConv2D(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
        #     norm_layer(width),
        #     ReLU())
        self.dilations = fluid.dygraph.container.LayerList()
        for dilation in dilations:
            self.dilations.append(
                fluid.dygraph.container.Sequential(
                    SeparableConv2D(3*width, width, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    norm_layer(width),
                    ReLU()
                )
            )

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].shape
        feats[-2] = fluid.layers.resize_bilinear(feats[-2], out_shape=(h, w))
        feats[-3] = fluid.layers.resize_bilinear(feats[-3], out_shape=(h, w))
        feat = fluid.layers.concat(feats, axis=1)
        feat = fluid.layers.concat(
            # [self.dilation1(feat),
            #  self.dilation2(feat),
            #  self.dilation3(feat),
            #  self.dilation4(feat)],
            [dilation(feat) for dilation in self.dilations],
            axis=1)
        return inputs[0], inputs[1], inputs[2], feat


if __name__ == '__main__':
    import numpy as np
    with fluid.dygraph.guard():
        conv5 = np.random.rand(4, 2048, 8, 8).astype("float32")
        conv4 = np.random.rand(4, 1024, 16, 16).astype("float32")
        conv3 = np.random.rand(4, 512, 32, 32).astype("float32")

        c5 = fluid.dygraph.to_variable(conv5)
        c4 = fluid.dygraph.to_variable(conv4)
        c3 = fluid.dygraph.to_variable(conv3)

        in_channels = [512, 1024, 2048]
        inputs = [c3, c4, c5]

        jpu = JPU(in_channels, norm_layer=fluid.dygraph.BatchNorm)
        out = jpu(*inputs)
        print(out[-1].shape)

        from PdSeg.models.modeling import JPU_X
        jpu_x = JPU_X(in_channels, norm_layer=fluid.dygraph.BatchNorm)
        out = jpu_x(*inputs)
        print(out[-1].shape)

