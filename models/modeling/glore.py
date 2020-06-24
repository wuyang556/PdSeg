# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/27
import paddle
import paddle.fluid as fluid
# from ..nn import ReLU
from PdSeg.models.nn import ReLU


class GCN(fluid.dygraph.Layer):
    """
    Graph Convolution Network.
    """
    def __init__(self, num_node, num_state, bias=False):
        super(GCN, self).__init__()
        self.linear1 = fluid.dygraph.Linear(num_node, num_node, bias_attr=bias)
        self.relu = ReLU()
        self.linear2 = fluid.dygraph.Linear(num_state, num_state, bias_attr=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = fluid.layers.transpose(x, [0, 2, 1])
        h = self.linear1(h)
        h = fluid.layers.transpose(h, [0, 2, 1])
        h = h + x

        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.linear2(h)
        return h


class Glore_Unit(fluid.dygraph.Layer):
    def __init__(self, num_in, num_mid, kernel=1, norm_layer=None):
        super(Glore_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.conv_state = fluid.dygraph.Conv2D(num_in, self.num_s, filter_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = fluid.dygraph.Conv2D(num_in, self.num_n, filter_size=kernel_size, padding=padding)
        # gcn
        self.gcn = GCN(num_state=self.num_n, num_node=self.num_s)
        # tail: extend dimension
        self.fc_2 = fluid.dygraph.Conv2D(self.num_s, num_in, filter_size=kernel_size, padding=padding, stride=(1, 1),
                                         groups=1, bias_attr=False)
        self.blocker = norm_layer(num_in)

    def forward(self, x):
        # x.shape = (n, c, h, w)
        batch_size, c, h, w = x.shape

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = fluid.layers.reshape(self.conv_state(x), shape=[batch_size, self.num_s, -1])

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = fluid.layers.reshape(self.conv_proj(x), shape=[batch_size, self.num_n, -1])

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = fluid.layers.matmul(x_state_reshaped, fluid.layers.transpose(x_proj_reshaped, [0, 2, 1]))
        x_n_state = x_n_state * (1. / x_state_reshaped.shape[2])

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        # print(x_n_state.shape)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = fluid.layers.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = fluid.layers.reshape(x_state_reshaped, shape=[batch_size, self.num_s, h, w])

        out = x + self.blocker(self.fc_2(x_state))

        return out


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        features = np.random.rand(4, 12, 64).astype("float32")
        gcn = GCN(12, 64)
        print(gcn.state_dict().keys())
        print(gcn.state_dict()['linear1.weight'].shape)
        print(gcn.state_dict()['linear2.weight'].shape)
        features = fluid.dygraph.to_variable(features)
        out = gcn(features)
        print(out.shape)

        glore = Glore_Unit(num_in=8, num_mid=64, norm_layer=fluid.dygraph.BatchNorm)
        in_np = np.random.rand(4, 8, 32, 32).astype("float32")
        in_var = fluid.dygraph.to_variable(in_np)
        out = glore(in_var)
        print(out.shape)

