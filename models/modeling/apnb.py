# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/4/19
import paddle.fluid as fluid
from ..nn import ReLU, Dropout2d, AdaptiveAvgPool2d, AdaptiveAvgPool3d


class PSPModule(fluid.dygraph.Layer):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = fluid.dygraph.container.LayerList(
            [self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 2:
            prior = AdaptiveAvgPool2d(pool_size=size, pool_type="avg")
        elif dimension == 3:
            prior = AdaptiveAvgPool3d(pool_size=size, pool_type="avg")
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.shape
        priors = [fluid.layers.reshape(stage(feats), shape=[n, c, -1]) for stage in self.stages]
        center = fluid.layers.concat(priors, axis=-1)
        return center


class _SelfAttentionBlock(fluid.dygraph.Layer):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_layer=fluid.dygraph.BatchNorm, psp_size=(1, 3, 6, 8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = fluid.dygraph.Pool2D(pool_size=scale, pool_type="max")
        self.f_key = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(num_channels=self.in_channels, num_filters=self.key_channels, filter_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            ReLU()
        )
        self.f_query = self.f_key
        self.f_value = fluid.dygraph.Conv2D(num_channels=self.in_channels, num_filters=self.value_channels, filter_size=1, stride=1, padding=0)
        self.W = fluid.dygraph.Conv2D(num_channels=self.value_channels, num_filters=self.out_channels, filter_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)

    def forward(self, x):
        batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]
        if self.scale > 1:
            x = self.pool(x)

        value = self.psp(self.f_value(x))

        query = fluid.layers.reshape(self.f_query(x), shape=(batch_size, self.key_channels, -1))
        query = fluid.layers.transpose(query, perm=[0, 2, 1])
        key = self.f_key(x)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = fluid.layers.transpose(value, perm=[0, 2, 1])
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = fluid.layers.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = fluid.layers.softmax(sim_map, axis=-1)

        context = fluid.layers.matmul(sim_map, value)
        context = fluid.layers.transpose(context, perm=[0, 2, 1])
        context = fluid.layers.reshape(context, shape=(batch_size, self.value_channels, x.shape[2], x.shape[3]))
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_layer=fluid.dygraph.BatchNorm, psp_size=(1, 3, 6, 8)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_layer,
                                                   psp_size=psp_size)


class APNB(fluid.dygraph.Layer):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), norm_layer=None, psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.stages = []
        self.norm_layer = norm_layer
        self.psp_size = psp_size
        self.stages = fluid.dygraph.container.LayerList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(num_channels=2 * in_channels, num_filters=out_channels, filter_size=1, padding=0),
            norm_layer(out_channels),
            ReLU(),
            Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_layer,
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(fluid.layers.concat([context, feats], axis=1))
        return output