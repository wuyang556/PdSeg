import paddle
import paddle.fluid as fluid

from .base import BaseNet
from .nn import ReLU, Dropout2d


class FCN(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, args, nclass, backbone, backbone_style, aux=True, se_loss=False, norm_layer=fluid.dygraph.BatchNorm, **kwargs):
        super(FCN, self).__init__(args, nclass, backbone, backbone_style, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCNHead(512*len(args.dilations), nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.shape[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = fluid.layers.resize_bilinear(input=x, out_shape=imsize)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = fluid.layers.resize_bilinear(input=auxout, out_shape=imsize)
            outputs.append(auxout)
        return tuple(outputs)

        
class FCNHead(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = fluid.dygraph.container.Sequential(fluid.dygraph.Conv2D(in_channels, inter_channels, 3, padding=1,
                                                                             bias_attr=False),
                                   norm_layer(inter_channels),
                                   ReLU(),
                                   Dropout2d(0.1),
                                   fluid.dygraph.Conv2D(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

