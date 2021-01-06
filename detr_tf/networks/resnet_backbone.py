import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, ReLU, MaxPool2D

from .custom_layers import FrozenBatchNorm2D


class ResNetBase(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pad1 = ZeroPadding2D(3, name='pad1')
        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='valid',
                            use_bias=False, name='conv1')
        self.bn1 = FrozenBatchNorm2D(name='bn1')
        self.relu = ReLU(name='relu')
        self.pad2 = ZeroPadding2D(1, name='pad2')
        self.maxpool = MaxPool2D(pool_size=3, strides=2, padding='valid')


    def call(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet50Backbone(ResNetBase):
    def __init__(self, replace_stride_with_dilation=[False, False, False], **kwargs):
        super().__init__(**kwargs)

        self.layer1 = ResidualBlock(num_bottlenecks=3, dim1=64, dim2=256, strides=1,
                                   replace_stride_with_dilation=False, name='layer1')
        self.layer2 = ResidualBlock(num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[0],
                                    name='layer2')
        self.layer3 = ResidualBlock(num_bottlenecks=6, dim1=256, dim2=1024, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[1],
                                    name='layer3')
        self.layer4 = ResidualBlock(num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[2],
                                    name='layer4')


class ResNet101Backbone(ResNetBase):
    def __init__(self, replace_stride_with_dilation=[False, False, False], **kwargs):
        super().__init__(**kwargs)

        self.layer1 = ResidualBlock(num_bottlenecks=3, dim1=64, dim2=256, strides=1,
                                    replace_stride_with_dilation=False, name='layer1')
        self.layer2 = ResidualBlock(num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[0],
                                    name='layer2')
        self.layer3 = ResidualBlock(num_bottlenecks=23, dim1=256, dim2=1024, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[1],
                                    name='layer3')
        self.layer4 = ResidualBlock(num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                                    replace_stride_with_dilation=replace_stride_with_dilation[2],
                                    name='layer4')


class ResidualBlock(tf.keras.Model):
    def __init__(self, num_bottlenecks, dim1, dim2, strides=1,
                 replace_stride_with_dilation=False, **kwargs):
        super().__init__(**kwargs)

        if replace_stride_with_dilation:
            strides = 1
            dilation = 2
        else:
            dilation = 1

        self.bottlenecks = [BottleNeck(dim1, dim2, strides=strides,
                                       downsample=True, name='0')]

        for idx in range(1, num_bottlenecks):
            self.bottlenecks.append(BottleNeck(dim1, dim2, name=str(idx),
                                               dilation=dilation))


    def call(self, x):
        for btn in self.bottlenecks:
            x = btn(x)
        return x


class BottleNeck(tf.keras.Model):
    def __init__(self, dim1, dim2, strides=1, dilation=1, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.downsample = downsample
        self.pad = ZeroPadding2D(dilation)
        self.relu = ReLU(name='relu')
        
        self.conv1 = Conv2D(dim1, kernel_size=1, use_bias=False, name='conv1')
        self.bn1 = FrozenBatchNorm2D(name='bn1')

        self.conv2 = Conv2D(dim1, kernel_size=3, strides=strides, dilation_rate=dilation,
                            use_bias=False, name='conv2')
        self.bn2 = FrozenBatchNorm2D(name='bn2')
        
        self.conv3 = Conv2D(dim2, kernel_size=1, use_bias=False, name='conv3')
        self.bn3 = FrozenBatchNorm2D(name='bn3')

        self.downsample_conv = Conv2D(dim2, kernel_size=1, strides=strides,
                                      use_bias=False, name='downsample_0')
        self.downsample_bn = FrozenBatchNorm2D(name='downsample_1')


    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out