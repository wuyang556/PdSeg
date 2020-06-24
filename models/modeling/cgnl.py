# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/4/18
import paddle
import paddle.fluid as fluid


class CGNL(fluid.dygraph.Layer):
    """
    compact non-local block
    CGNL block with dot production kernel for image classification.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=8):
        super(CGNL, self).__init__()
        self.use_scale = use_scale
        self.groups = groups

        # conv theta
        self.t = fluid.dygraph.Conv2D(num_channels=inplanes, num_filters=planes, filter_size=1, stride=1, bias_attr=False)
        # conv phi
        self.p = fluid.dygraph.Conv2D(num_channels=inplanes, num_filters=planes, filter_size=1, stride=1, bias_attr=False)
        # conv g
        self.g = fluid.dygraph.Conv2D(num_channels=inplanes, num_filters=planes, filter_size=1, stride=1, bias_attr=False)
        # conv z
        self.z = fluid.dygraph.Conv2D(num_channels=planes, num_filters=inplanes, filter_size=1, stride=1, bias_attr=False, groups=groups)

        self.gn = fluid.dygraph.GroupNorm(channels=inplanes, groups=groups)

    def kernel(self, t, p, g, b, c, h, w):
        """
        The linear kernel(dot production)

        :param t: output of conv theata
        :param p: output of conv phi
        :param g: output of conv g
        :param b: batch size
        :param c: channels number
        :param h: height of feature maps
        :param w: width of feature maps
        :return:
        """
        # print(t.shape)
        # print(b, c, h, w)
        t = fluid.layers.reshape(t, shape=[b, 1, c*h*w])
        p = fluid.layers.reshape(p, shape=[b, 1, c*h*w])
        g = fluid.layers.reshape(g, shape=[b, c*h*w, 1])

        att = fluid.layers.matmul(p, g)

        if self.use_scale:
            att = fluid.layers.elementwise_div(att, (c*h*w)**0.5)

        x = fluid.layers.matmul(att, t)
        x = fluid.layers.reshape(x, shape=[b, c, h, w])

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.shape

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = fluid.layers.split(t, num_or_sections=self.groups, dim=1)
            ps = fluid.layers.split(p, num_or_sections=self.groups, dim=1)
            gs = fluid.layers.split(g, num_or_sections=self.groups, dim=1)
            # print(len(ts), _c, c)
            # print(ts[0].shape)

            _t_sequences = []

            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = fluid.layers.concat(_t_sequences, axis=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual
        return x


if __name__ == '__main__':
    import numpy as np
    in_np = np.ones(shape=[4, 512, 32, 32], dtype="float32")
    with fluid.dygraph.guard():
        in_var = fluid.dygraph.to_variable(in_np)
        model = CGNL(inplanes=512, planes=256)
        out = model(in_var)
        print(out.shape)