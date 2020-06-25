## 基于Paddle Paddle动态图的图像语义分割工具库

### 介绍：

1. 由于之前PaddlePaddle提供的图像识别模型都是以静态图形式保存，在其动态图模式下不易重复使用，于是使用迁移torchvision中相关模型数据。本工具库中所有模型使用的backbone均使用paddlevision项目中的模型参数，链接：https://github.com/wuyang556/paddlevision 
2. 使用PaddlePaddle动态图模式设计出一个专门用于图像语义分割的模型库，主要功能模块包括有数据集加载器reader(主要数据集有PASCAL VOC2012，COCO2017，PASCAL Context，Cityscapes，Ade20k)，模型backbone(使用主流图像识别模 型如vgg-net，resnet，以及resnest)，分割模型(FCN，PSPNet，DeepLabV3)，基于上述模块，可以使用PaddlePaddle的 dygraph轻松设计新的分割模型快速开始实验。 
3. 以上所提的相关数据集均已在Baidu AiStudio开放，使用AiStudio创建项目时，可以直接引用，也可使用dataset文件夹中的相应代码下载voc， cityscapes，pet等数据集。
4. 在数据集上使用了一些数据增强的方法，如翻转角度，平移，镜像等。
5. 可直接使用本工具库，通过继承base.py中的模型基类，继续设计自己的模型，只专注于设计模型，后续过程，不用重复，极大节省开发时间。

### 主要目录说明：

1. dataset：下载相关数据集如voc2012，cityscapes，pet等。
2. reader：数据集加载器。
3. models：涉及到的主要模型代码，如fcn，pspnet，deeplabv3等。
4. utils：辅助功能代码，包含了交叉熵损失函数定义，模型快照保存，主要评价标准pixAcc，mIoU，整个训练过程结果保存，简单训练可视化等功能。

### 使用方式：

- 下载使用的backnone的预训练数据，然后放在对应的模型代码目录之下。

- 先在option.py配置相关参数，选择模型和backbone，设置相关超参数。
- 然后运行train.py文件，即可训练模型。

### 参考代码：

- Pytorch-Encoding
- PaddleSeg

