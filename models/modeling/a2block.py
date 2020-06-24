# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/4/18

import paddle
import paddle.fluid as fluid
from ..nn import ReLU, Softmax
# from PdSeg.models.nn import Softmax


class A2Block(fluid.dygraph.Layer):
    """
    A2-Net
    """
    def __init__(self, inplane, plane):
        super(A2Block, self).__init__()
        self.down = fluid.dygraph.Conv2D(num_channels=inplane, num_filters=plane, filter_size=1)
        self.up = fluid.dygraph.Conv2D(num_channels=plane, num_filters=inplane, filter_size=1)
        self.gather_down = fluid.dygraph.Conv2D(num_channels=inplane, num_filters=plane, filter_size=1)
        self.distribute_down = fluid.dygraph.Conv2D(num_channels=inplane, num_filters=plane, filter_size=1)
        self.softmax = Softmax(axis=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, w = A.shape
        A = fluid.layers.reshape(A, shape=[b, c, -1])  # (b, c, h*w)
        B = fluid.layers.reshape(A, shape=[b, c, -1])  # (b, c, h*w)
        B = self.softmax(B)
        B = fluid.layers.transpose(B, perm=[0, 2, 1])  # (b, h*w, c)

        G = fluid.layers.matmul(A, B)  # (b, c, c)

        C = self.distribute_down(res)
        C = fluid.layers.reshape(C, shape=[b, c, -1])  # (b, c, h*w)
        C = self.softmax(C)
        C = fluid.layers.transpose(C, perm=[0, 2, 1])  # (b, h*w, c)

        atten = fluid.layers.matmul(C, G)  # (b, h*w, c)
        atten = fluid.layers.transpose(atten, perm=[0, 2, 1])  # (b, c, h*w)
        atten = fluid.layers.reshape(atten, shape=[b, c, h, w])
        atten = self.up(atten)

        out = res + atten
        return out


if __name__ == '__main__':
    import numpy as np
    in_np = np.ones(shape=[4, 512, 32, 32], dtype="float32")
    with fluid.dygraph.guard():
        in_var = fluid.dygraph.to_variable(in_np)
        model = A2Block(inplane=512, plane=256)
        out = model(in_var)
        print(out.shape)
