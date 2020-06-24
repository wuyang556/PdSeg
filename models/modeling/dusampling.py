# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/5/30
import paddle.fluid as fluid


class DUpsampling(fluid.dygraph.Layer):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = fluid.dygraph.Conv2D(in_channels, out_channels * scale_factor * scale_factor, 1, bias_attr=False)

    def forward(self, x):
        x = self.conv_w(x)
        # n, c, h, w = x.size()
        n, c, h, w = x.shape

        # N, C, H, W --> N, W, H, C
        # x = x.permute(0, 3, 2, 1).contiguous()
        x = fluid.layers.transpose(x, perm=[0, 3, 2, 1])

        # N, W, H, C --> N, W, H * scale, C // scale
        # x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)
        x = fluid.layers.reshape(x, shape=(n, w, h * self.scale_factor, c // self.scale_factor))

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        # x = x.permute(0, 2, 1, 3).contiguous()
        x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        # x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))
        x = fluid.layers.reshape(x, shape=(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor)))

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        # x = x.permute(0, 3, 1, 2)
        x = fluid.layers.transpose(x, perm=[0, 3, 1, 2])

        return x


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np

        in_np = np.ones(shape=(4, 512, 32, 32), dtype="float32")
        in_var = fluid.dygraph.to_variable(in_np)
        model = DUpsampling(512, 21)
        out = model(in_var)
        print(out.shape)
