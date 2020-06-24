import math
import numpy as np


import paddle
import paddle.fluid as fluid

# from ..nn import JPU
# from .. import dilated as resnet
# from ..utils import batch_pix_accuracy, batch_intersection_union
from PdSeg.models import dilated
from PdSeg.models import backbone as bk
from PdSeg.models.dilated import resnet, resnest
from PdSeg.utils.metrics import batch_intersection_union, batch_pix_accuracy
from PdSeg.models.modeling import JPU, JPU_X


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet']


class BaseNet(fluid.dygraph.Layer):
    def __init__(self, args, nclass, backbone, backbone_style, aux, se_loss, pu="jpu", dilated=False, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/work/PdSeg/models', **kwargs):
        super(BaseNet, self).__init__()
        self.args = args
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            if backbone_style == "pytorch":
                self.pretrained = bk.resnet50(pretrained=True)
            else:
                raise RuntimeError('unknown style: {0}'.format(backbone_style))
        elif backbone == 'resnet101':
            if backbone_style == "pytorch":
                self.pretrained = bk.resnet101(pretrained=True)
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'resnet152':
            if backbone_style == "pytorch":
                self.pretrained = bk.resnet152(pretrained=True)
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'resnest50':
            if backbone_style == "pytorch":
                raise NotImplemented(f"The pytorch backbone style of {backbone_style} is not implemented.")
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'resnest101':
            if backbone_style == "pytorch":
                raise NotImplemented(f"The pytorch backbone style of {backbone_style} is not implemented.")
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'resnest200':
            if backbone_style == "pytorch":
                raise NotImplemented(f"The pytorch backbone style of {backbone_style} is not implemented.")
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'resnest269':
            if backbone_style == "pytorch":
                raise NotImplemented(f"The pytorch backbone style of {backbone_style} is not implemented.")
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'res2net50':
            if backbone_style == "pytorch":
                self.pretrained = bk.res2net50_v1b(pretrained=True, dilated=dilated, norm_layer=norm_layer)

            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'res2net101':
            if backbone_style == "pytorch":
                self.pretrained = bk.res2net101_v1b(pretrained=True, dilated=dilated, norm_layer=norm_layer)

            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'res2next50':
            if backbone_style == "pytorch":
                self.pretrained = bk.res2next50(pretrained=True, dilated=dilated, norm_layer=norm_layer)

            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'scnet50':
            if backbone_style == "pytorch":
                self.pretrained = bk.scnet50(pretrained=True, dilated=dilated, norm_layer=norm_layer)
            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        elif backbone == 'scnet101':
            if backbone_style == "pytorch":
                self.pretrained = bk.scnet101(pretrained=True, dilated=dilated, norm_layer=norm_layer)

            else:
                raise RuntimeError('unknown style: {}'.format(backbone_style))
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.backbone = backbone
        self.pu = pu
        if pu == "jpu":
            self.up = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, dilations=args.dilations)
        elif pu == "jpu_x":
            self.up = JPU_X([512, 1024, 2048], width=512, norm_layer=norm_layer)
        elif pu is None:
            self.up = fluid.dygraph.container.Sequential()
        else:
            raise RuntimeError(f"The {pu} is wrong.")

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.pu:
            return self.up(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred, target)
        inter, union = batch_intersection_union(pred, target, self.nclass)
        return correct, labeled, inter, union


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = BaseNet(nclass=21, backbone="resnet50", style="th", aux=False, se_loss=False, norm_layer=fluid.dygraph.BatchNorm)
        in_np = np.random.rand(4, 3, 224, 224).astype("float32")
        in_var = fluid.dygraph.to_variable(in_np)
        out = model.base_forward(in_var)
        print(out[0].shape)
        print(out[1].shape)
        print(out[2].shape)
        print(out[3].shape)


