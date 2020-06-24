# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/4/18
import paddle
import paddle.fluid as fluid
# from ..nn import AdaptiveAvgPool2d, ReLU, Sigmoid
from PdSeg.models.nn import AdaptiveAvgPool2d, ReLU, Sigmoid


class CBAM_Module(fluid.dygraph.Layer):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(pool_size=1, pool_type="avg")
        self.max_pool = AdaptiveAvgPool2d(pool_size=1, pool_type="max")
        self.fc1 = fluid.dygraph.Conv2D(num_channels=channels, num_filters=channels // reduction, filter_size=1, padding=0)
        self.relu = ReLU()
        self.fc2 = fluid.dygraph.Conv2D(num_channels=channels // reduction, num_filters=channels, filter_size=1, padding=0)

        self.sigmoid_channel = Sigmoid()
        self.conv_after_concat = fluid.dygraph.Conv2D(num_channels=2, num_filters=1, filter_size=7, stride=1, padding=3)
        self.sigmoid_spatial = Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        module_input = x
        avg = self.relu(self.fc1(self.avg_pool(x)))
        avg = self.fc2(avg)
        mx = self.relu(self.fc1(self.max_pool(x)))
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial Attention Module
        x = module_input * x
        module_input = x
        avg = fluid.layers.mean(x)
        mx = fluid.layers.argmax(x, axis=1)
        print(avg.shape, mx.shape)
        x = fluid.layers.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x

        return x


if __name__ == '__main__':
    import numpy as np
    in_np = np.ones(shape=[4, 512, 32, 32], dtype="float32")
    with fluid.dygraph.guard():
        in_var = fluid.dygraph.to_variable(in_np)
        model = CBAM_Module(channels=512, reduction=8)
        out = model(in_var)
        print(out.shape)
