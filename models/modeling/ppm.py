# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24
import paddle
import paddle.fluid as fluid
from PdSeg.models.nn import ReLU, Dropout2d, AdaptiveAvgPool2d


class PyramidPooling(fluid.dygraph.Layer):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv2 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv3 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv4 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())

    def forward(self, x):
        _, _, h, w = x.shape
        feat1 = fluid.layers.resize_bilinear(self.conv1(self.pool1(x)), out_shape=(h, w))
        feat2 = fluid.layers.resize_bilinear(self.conv2(self.pool2(x)), out_shape=(h, w))
        feat3 = fluid.layers.resize_bilinear(self.conv3(self.pool3(x)), out_shape=(h, w))
        feat4 = fluid.layers.resize_bilinear(self.conv4(self.pool4(x)), out_shape=(h, w))
        return fluid.layers.concat([x, feat1, feat2, feat3, feat4], 1)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        in_np = np.random.rand(4, 2048, 8, 8).astype("float32")
        in_var = fluid.dygraph.to_variable(in_np)
        psp = PyramidPooling(in_channels=2048, norm_layer=fluid.dygraph.BatchNorm)
        out = psp(in_var)
        print(out.shape)
