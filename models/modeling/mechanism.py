# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/4/16

import paddle.fluid as fluid


class PAM(fluid.dygraph.Layer):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        inter_channels = in_channels // 8
        self.conv_b = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=inter_channels, filter_size=1, bias_attr=False)
        self.conv_c = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=inter_channels, filter_size=1, bias_attr=False)
        self.conv_d = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=in_channels, filter_size=1, bias_attr=False)

    def forward(self, x):
        N, C, H, W = x.shape
        b = fluid.layers.transpose(fluid.layers.reshape(self.conv_b(x), shape=[N, -1, H*W]), perm=[0, 2, 1])
        c = fluid.layers.reshape(self.conv_c(x), shape=[N, -1, H*W])
        energy = fluid.layers.softmax(fluid.layers.matmul(b, c))
        d = fluid.layers.reshape(self.conv_d(x), shape=[N, C, H*W])
        attention = fluid.layers.reshape(fluid.layers.matmul(d, energy), shape=[N, C, H, W])
        out = attention + x
        return out


class CAM(fluid.dygraph.Layer):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        inter_channels = in_channels // 8
        self.conv_b = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=in_channels, filter_size=1, bias_attr=False)
        self.conv_c = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=in_channels, filter_size=1, bias_attr=False)
        self.conv_d = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=in_channels, filter_size=1, bias_attr=False)

    def forward(self, x):
        N, C, H, W = x.shape
        b = fluid.layers.transpose(fluid.layers.reshape(self.conv_b(x), shape=[N, C, H*W]), perm=[0, 2, 1])
        c = fluid.layers.reshape(self.conv_c(x), shape=[N, C, H*W])
        energy = fluid.layers.softmax(fluid.layers.matmul(c, b), axis=-1)
        d = fluid.layers.reshape(self.conv_d(x), shape=[N, C, H*W])
        attention = fluid.layers.matmul(energy, d)
        attention = fluid.layers.reshape(attention, shape=[N, C, H, W])
        out = attention + x
        return out




if __name__ == '__main__':
    import numpy as np
    in_np = np.zeros(shape=[4, 512, 32, 32], dtype="float32")
    with fluid.dygraph.guard():
        in_var = fluid.dygraph.to_variable(in_np)
        pam = PAM(in_channels=512)
        out = pam(in_var)
        cam = CAM(in_channels=512)
        out = cam(in_var)
        print(out.shape)
