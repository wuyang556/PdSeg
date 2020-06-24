# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24
from .base import BaseNet
from .fcn import FCN, FCNHead
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3


models = {
    "fcn": FCN,
    "pspnet": PSPNet,
    "deeplabv3": DeepLabV3,
}


def get_segmentation_model(name, **kwargs):
    return models[name.lower()](**kwargs)
