from __future__ import division
import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet import cpu
from gluoncv.model_zoo.cifarresnet import _get_resnet_spec
import os
import mxnet as mx
from mxnet import gluon
from mxnet.initializer import Xavier
from gluoncv.model_zoo.ssd.vgg_atrous import Normalize
from gluoncv.model_zoo.ssd.vgg_atrous import vgg_spec, extra_spec


class LearnedCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1,2)):
        super(LearnedCell, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.feature_channel = nn.HybridSequential(prefix='feature_channel')
            self.feature_channel.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                                               padding=0))
            self.feature_channel.add(nn.BatchNorm())
            self.feature_channel.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xc = F.concat(x3, x5, dim=1)
        xc = self.feature_channel(xc)

        return xc


class LearnedConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1):
        super(LearnedConv, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.feature_channel = nn.HybridSequential(prefix='feature_channel')
            self.feature_channel.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                                               padding=0))

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xc = F.concat(x3, x5, dim=1)
        xc = self.feature_channel(xc)

        return xc


class ChaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2)):
        super(ChaDyReFCell, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = F.concat(x3, x5, dim=1)
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class ChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1):
        super(ChaDyReFConv, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = F.concat(x3, x5, dim=1)
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        return xs


class SKCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2)):
        super(SKCell, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)
        self.softmax_channels = int(channels * 2)

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())
            self.feature_spatial_3.add(nn.Activation('relu'))

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())
            self.feature_spatial_5.add(nn.Activation('relu'))

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        xa = F.softmax(xa, axis=1)

        xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        xa3 = F.reshape(xa3, (0, -1, 1, 1))
        xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xs = x3 * xa3 + x5 * xa5

        return xs



class SKConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1):
        super(SKConv, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)
        self.softmax_channels = int(channels * 2)

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            # self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
            #                                      padding=0, dilation=1))
            self.feature_spatial_3.add(nn.BatchNorm())
            # self.feature_spatial_3.add(nn.Activation('relu'))

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            # self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
            #                                      padding=1, dilation=1))
            self.feature_spatial_5.add(nn.BatchNorm())
            # self.feature_spatial_5.add(nn.Activation('relu'))

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        xa = F.softmax(xa, axis=1)

        xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        xa3 = F.reshape(xa3, (0, -1, 1, 1))
        xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xs = F.broadcast_mul(x3, xa3) + F.broadcast_mul(x5, xa5)
        # xs = x3 * xa3 + x5 * xa5

        return xs


class SK_ChaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2)):
        super(SK_ChaDyReFCell, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_ChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, useGlobal=False):
        super(SK_ChaDyReFConv, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            if useGlobal:
                self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        # xs = x3 * xa + x5 * (1 - xa)
        xs = F.broadcast_mul(x3, xa) + F.broadcast_mul(x5, 1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SK_SpaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, asBackbone=False):
        super(SK_SpaDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            if self.asBackbone:
                self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                             padding=act_dilation, dilation=act_dilation,
                                             groups=channels))
            else:
                self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                             padding=act_dilation, dilation=act_dilation,
                                             groups=channels))
                self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        if self.asBackbone:
            xs = self.attention(xa)
        else:
            xa = self.attention(xa)
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_SpaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, asBackbone=False):
        super(SK_SpaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            if self.asBackbone:
                self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                             padding=act_dilation, dilation=act_dilation,
                                             groups=channels))
            else:
                self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                             padding=act_dilation, dilation=act_dilation,
                                             groups=channels))
                self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        if self.asBackbone:
            xs = self.attention(xa)
        else:
            xa = self.attention(xa)
            xs = x3 * xa + x5 * (1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SK_1x1DepthDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2)):
        super(SK_1x1DepthDyReFCell, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_1x1DepthDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1):
        super(SK_1x1DepthDyReFConv, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
            #                              groups=channels))
            # self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))

            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SK_MSSpaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=('xxx', 'xxx'),
                 asBackbone=False):
        super(SK_MSSpaDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone
        dilation_1 = dilations[0]
        dilation_2 = dilations[1]
        act_dilation_1 = act_dilation[0]
        act_dilation_2 = act_dilation[1]

        with self.name_scope():

            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            # self.attention_4 = nn.HybridSequential(prefix='attention_4')
            # self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
            #                                dilation=4, groups=channels))
            # self.attention_4.add(nn.BatchNorm())

            self.attention_8 = nn.HybridSequential(prefix='attention_8')
            self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                           padding=act_dilation_1,
                                           dilation=act_dilation_1, groups=channels))
            self.attention_8.add(nn.BatchNorm())

            self.attention_16 = nn.HybridSequential(prefix='attention_16')
            self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                            padding=act_dilation_2,
                                            dilation=act_dilation_2, groups=channels))
            self.attention_16.add(nn.BatchNorm())

            # self.attention_32 = nn.HybridSequential(prefix='attention_32')
            # self.attention_32.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=32,
            #                                dilation=32, groups=channels))
            # self.attention_32.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        # xa4 = self.attention_4(xa)
        xa8 = self.attention_8(xa)
        xa16 = self.attention_16(xa)
        # xa32 = self.attention_32(xa)
        # xa = xa4 + xa8 + xa16
        # xa = xa8 + xa16 + xa32
        xa = xa8 + xa16

        if self.asBackbone:
            xs = xa
        else:
            xa = self.sigmoid(xa)
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs




class iAAMSSpaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False):
        super(iAAMSSpaDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.att2_4 = nn.HybridSequential(prefix='att2_4')
            self.att2_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
                                      dilation=4, groups=channels))
            self.att2_4.add(nn.BatchNorm())

            self.att2_8 = nn.HybridSequential(prefix='att2_8')
            self.att2_8.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=8,
                                      dilation=8, groups=channels))
            self.att2_8.add(nn.BatchNorm())

            self.att2_16 = nn.HybridSequential(prefix='att2_16')
            self.att2_16.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=16,
                                       dilation=16, groups=channels))
            self.att2_16.add(nn.BatchNorm())

            self.channel_att = nn.HybridSequential(prefix='channel_att')
            self.channel_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.channel_att.add(nn.BatchNorm())
            self.channel_att.add(nn.Activation('sigmoid'))

            # self.sigmoid2 = nn.HybridSequential(prefix='sigmoid2')
            # self.sigmoid2.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xa = x3 + x5

        xa24 = self.att2_4(xa)
        xa28 = self.att2_8(xa)
        xa216 = self.att2_16(xa)
        xa3 = xa24 + xa28 + xa216

        attwei = self.channel_att(xa3)
        xs = x3 * attwei + x5 * (1 - attwei)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs



class SK_MSSeqDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False):
        super(SK_MSSeqDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            # self.attention_4 = nn.HybridSequential(prefix='attention_4')
            # self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
            #                                dilation=4, groups=channels))
            # self.attention_4.add(nn.BatchNorm())

            self.attention_8 = nn.HybridSequential(prefix='attention_8')
            self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=8,
                                           dilation=8, groups=channels))
            self.attention_8.add(nn.BatchNorm())

            self.attention_16 = nn.HybridSequential(prefix='attention_16')
            self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=16,
                                           dilation=16, groups=channels))
            self.attention_16.add(nn.BatchNorm())

            self.attention_32 = nn.HybridSequential(prefix='attention_32')
            self.attention_32.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=32,
                                           dilation=32, groups=channels))
            self.attention_32.add(nn.BatchNorm())

            self.channel = nn.HybridSequential(prefix='channel')
            self.channel.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.channel.add(nn.BatchNorm())
            # self.channel.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        # xa4 = self.attention_4(xa)
        xa8 = self.attention_8(xa)
        xa16 = self.attention_16(xa)
        xa32 = self.attention_32(xa)
        # xa = xa4 + xa8 + xa16
        xa = xa8 + xa16 + xa32
        xa = self.channel(xa)
        # xa = xa8 + xa16

        if self.asBackbone:
            xs = xa
        else:
            xa = self.sigmoid(xa)
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs



class Sub_MSSpaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False):
        super(Sub_MSSpaDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            # self.attention_4 = nn.HybridSequential(prefix='attention_4')
            # self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
            #                                dilation=4, groups=channels))
            # self.attention_4.add(nn.BatchNorm())

            self.attention_8 = nn.HybridSequential(prefix='attention_8')
            self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=8,
                                           dilation=8, groups=channels))
            self.attention_8.add(nn.BatchNorm())

            self.attention_16 = nn.HybridSequential(prefix='attention_16')
            self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=16,
                                           dilation=16, groups=channels))
            self.attention_16.add(nn.BatchNorm())

            self.attention_32 = nn.HybridSequential(prefix='attention_32')
            self.attention_32.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=32,
                                           dilation=32, groups=channels))
            self.attention_32.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 - x5
        # xa4 = self.attention_4(xa)
        xa8 = self.attention_8(xa)
        xa16 = self.attention_16(xa)
        xa32 = self.attention_32(xa)
        # xa = xa4 + xa8 + xa16
        xa = xa8 + xa16 + xa32
        # xa = xa8 + xa16

        if self.asBackbone:
            xs = xa
        else:
            xa = self.sigmoid(xa)
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_MSSpaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False, stride=1):
        super(SK_MSSpaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention_4 = nn.HybridSequential(prefix='attention_4')
            self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
                                           dilation=4, groups=channels))
            self.attention_4.add(nn.BatchNorm())

            self.attention_8 = nn.HybridSequential(prefix='attention_8')
            self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=8,
                                           dilation=8, groups=channels))
            self.attention_8.add(nn.BatchNorm())

            self.attention_16 = nn.HybridSequential(prefix='attention_16')
            self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=16,
                                           dilation=16, groups=channels))
            self.attention_16.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa4 = self.attention_4(xa)
        xa8 = self.attention_8(xa)
        xa16 = self.attention_16(xa)
        xa = xa4 + xa8 + xa16
        # xa = xa8 + xa16

        if self.asBackbone:
            xs = xa
        else:
            xa = self.sigmoid(xa)
            xs = x3 * xa + x5 * (1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class Direct_AddCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False):
        super(Direct_AddCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xs = x3 + x5
        xs = self.relu(xs)

        return xs


class Direct_AddConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), asBackbone=False, stride=1):
        super(Direct_AddConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            # self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            # self.feature_spatial_5.add(nn.BatchNorm())

            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xs = x3 + x5
        # xs = self.relu(xs)

        return xs


class SK_SpaDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1):
        super(SK_SpaDyReFCell, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         padding=act_dilation, dilation=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        xa = self.sigmoid(xa)

        xs = x3 * xa + x5 * (1 - xa)
        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_SpaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, stride=1):
        super(SK_SpaDyReFConv, self).__init__()
        self.channels = channels

        with self.name_scope():

            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         padding=act_dilation, dilation=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5
        xa = self.attention(xa)
        xa = self.sigmoid(xa)

        xs = x3 * xa + x5 * (1 - xa)
        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SeqDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SeqDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            if useReLU:
                self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         dilation=act_dilation, padding=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            if useReLU:
                self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            if not asBackbone:
                self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = F.concat(x3, x5, dim=1)
        xa = self.attention(xa)

        if self.asBackbone:
            xs = xa
        else:
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SK_SeqDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_SeqDyReFCell, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.attention.add(nn.BatchNorm())
            # if useReLU:
            #     self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         dilation=act_dilation, padding=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            if useReLU:
                self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            if not asBackbone:
                self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        # xa = F.concat(x3, x5, dim=1)
        xa = x3 + x5
        xa = self.attention(xa)

        if self.asBackbone:
            xs = xa
        else:
            xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs


class SeqDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SeqDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.attention.add(nn.BatchNorm())
            # if useReLU:
            #     self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         dilation=act_dilation, padding=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            # if useReLU:
            #     self.attention.add(nn.Activation('relu'))
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            if not asBackbone:
                # self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = F.concat(x3, x5, dim=1)
        xa = self.attention(xa)

        if self.asBackbone:
            xs = xa
        else:
            xs = x3 * xa + x5 * (1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SK_SeqDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_SeqDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.attention.add(nn.BatchNorm())
            # if useReLU:
            #     self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         dilation=act_dilation, padding=act_dilation, groups=channels))
            self.attention.add(nn.BatchNorm())
            # self.attention.add(nn.Activation('relu'))
            # if useReLU:
            #     self.attention.add(nn.Activation('relu'))
            # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            if not asBackbone:
                # self.attention.add(nn.BatchNorm())
                self.attention.add(nn.Activation('sigmoid'))

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        # xa = F.concat(x3, x5, dim=1)
        xa = x3 + x5
        xa = self.attention(xa)

        if self.asBackbone:
            xs = xa
        else:
            xs = x3 * xa + x5 * (1 - xa)

        # xs = self.bn(xs)
        # xs = self.relu(xs)

        return xs


class SK_LGChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_LGChaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            # self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5                  # input of attention
        xl = self.local_att(xa)       # local attention
        xg = self.global_att(xa)      # global attention
        xo = F.broadcast_add(xl, xg)  # output of attention
        xa = self.sigmoid(xo)

        xs = x3 * xa + x5 * (1 - xa)

        return xs



class AYforXplusYDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(AYforXplusYDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            # self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x5                  # input of attention
        xl = self.local_att(xa)       # local attention
        xg = self.global_att(xa)      # global attention
        xo = F.broadcast_add(xl, xg)  # output of attention
        xa = self.sigmoid(xo)

        xs = x3 * xa + x5

        return xs



class XplusAYforYDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(XplusAYforYDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            # self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x5                  # input of attention
        xl = self.local_att(xa)       # local attention
        xg = self.global_att(xa)      # global attention
        xo = F.broadcast_add(xl, xg)  # output of attention
        xa = self.sigmoid(xo)

        xs = x3 + x5 * xa

        return xs



class AXYforXplusYDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(AXYforXplusYDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            # self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5                  # input of attention
        xl = self.local_att(xa)       # local attention
        xg = self.global_att(xa)      # global attention
        xo = F.broadcast_add(xl, xg)  # output of attention
        xa = self.sigmoid(xo)

        xs = x3 * xa + x5

        return xs



class MatryoshkaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(MatryoshkaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local1 = nn.HybridSequential(prefix='local1')
            self.local1.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local1.add(nn.BatchNorm())

            self.global1 = nn.HybridSequential(prefix='global1')
            self.global1.add(nn.GlobalAvgPool2D())
            self.global1.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global1.add(nn.BatchNorm())

            self.local2 = nn.HybridSequential(prefix='local2')
            self.local2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local2.add(nn.BatchNorm())

            self.global2 = nn.HybridSequential(prefix='global2')
            self.global2.add(nn.GlobalAvgPool2D())
            self.global2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global2.add(nn.BatchNorm())

            self.sig1 = nn.Activation('sigmoid')
            self.sig2 = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa1 = x3 + x5                  # input of attention
        xl1 = self.local1(xa1)       # local attention
        xg1 = self.global1(xa1)      # global attention
        xf1 = F.broadcast_add(xl1, xg1)  # output of attention
        wei1 = self.sig1(xf1)

        xa2 = x3 * wei1 + x5 * (1 - wei1)
        xl2 = self.local2(xa2)
        xg2 = self.global2(xa2)
        xf2 = F.broadcast_add(xl2, xg2)
        wei2 = self.sig1(xf2)

        xo = x3 * wei2 + x5 * (1 - wei2)

        return xo


class SK_LadderLGChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_LadderLGChaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            # self.local_att.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            # self.global_att.add(nn.Activation('relu'))

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = x3 + x5                  # input of attention
        xl = self.local_att(xa)       # local attention
        xg = self.global_att(xl + xa)      # global attention
        xo = F.broadcast_add(xl, xg)  # output of attention
        xa = self.sigmoid(xo)

        xs = x3 * xa + x5 * (1 - xa)

        return xs


class SK_AAPChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_AAPChaDyReFConv, self).__init__()
        self.channels = channels
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            # mx.nd.contrib.AdaptiveAvgPooling2D()
            # F.contrib.AdaptiveAvgPooling2D()
            self.aap_att = nn.HybridSequential(prefix='aap_att')
            self.aap_att.add(nn.AvgPool2D(pool_size=(5, 5), strides=1, padding=2))
            self.aap_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.aap_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):


        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)
        xi = x3 + x5                  # input of attention
        # _, _, hei, wid = xi.shape

        # xa = F.contrib.AdaptiveAvgPooling2D(xi, (2, 2))
        xa = self.aap_att(xi)
        wei = self.sigmoid(xa)
        # wei = F.UpSampling(wei, scale=int(hei//2), sample_type='nearest')

        xo = F.broadcast_mul(x3, wei) + F.broadcast_mul(x5, 1 - wei)

        # xl = self.local_att(xa)       # local attention
        # xg = self.global_att(xl + xa)      # global attention
        # xo = F.broadcast_add(xl, xg)  # output of attention
        # xa = self.sigmoid(xo)
        #
        # xs = x3 * xa + x5 * (1 - xa)

        return xo



class SK_TwoLGChaDyReFConv(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
                 asBackbone=False):
        super(SK_TwoLGChaDyReFConv, self).__init__()
        self.channels = channels
        self.inter_channels = int(channels // 4)
        self.asBackbone = asBackbone

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            # self.local_att.add(nn.Activation('relu'))
            # self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1, padding=0))
            # self.local_att.add(nn.BatchNorm())
            # self.local_att.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))

            self.sec_local_att = nn.HybridSequential(prefix='sec_local_att')
            self.sec_local_att.add(nn.Conv2D(self.inter_channels, kernel_size=1, strides=1, padding=0))
            self.sec_local_att.add(nn.BatchNorm())
            self.sec_local_att.add(nn.Activation('relu'))
            self.sec_local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1, padding=0))
            self.sec_local_att.add(nn.BatchNorm())

            self.sec_global_att = nn.HybridSequential(prefix='sec_global_att')
            self.sec_global_att.add(nn.GlobalAvgPool2D())
            self.sec_global_att.add(nn.Conv2D(self.inter_channels, kernel_size=1, strides=1, padding=0))
            self.sec_global_att.add(nn.BatchNorm())
            self.sec_global_att.add(nn.Activation('relu'))
            self.sec_global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1, padding=0))
            self.sec_global_att.add(nn.BatchNorm())

            # self.sigmoid = nn.Activation('sigmoid')
            # self.relu = nn.Activation('relu')
            self.sigmoid2 = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa1 = x3 + x5                  # input of attention
        xl1 = self.local_att(xa1)       # local attention
        xg1 = self.global_att(xa1)      # global attention
        # xo1 = F.broadcast_add(xl1, xg1)  # output of attention
        # xa2 = self.relu(xo1)
        xa2 = F.broadcast_add(xl1, xg1)  # output of attention

        # wei1 = self.sigmoid(xo1)
        # xa2 = x3 * wei1 + x5 * (1 - wei1)

        xl2 = self.sec_local_att(xa2)       # local attention
        xg2 = self.sec_global_att(xa2)      # global attention
        xo2 = F.broadcast_add(xl2, xg2)  # output of attention
        wei2 = self.sigmoid2(xo2)
        xs = x3 * wei2 + x5 * (1 - wei2)

        return xs


class MSSeqDyReFCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2), act_dilation=(1, 2)):
        super(MSSeqDyReFCell, self).__init__()
        self.channels = channels
        inter_channels = channels // 2

        with self.name_scope():
            dilation_1 = dilations[0]
            dilation_2 = dilations[1]
            act_dilation_1 = act_dilation[0]
            act_dilation_2 = act_dilation[1]
            self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
            self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_1, dilation=dilation_1))
            self.feature_spatial_3.add(nn.BatchNorm())

            self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
            self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dilation_2, dilation=dilation_2))
            self.feature_spatial_5.add(nn.BatchNorm())

            self.pre_attention = nn.HybridSequential(prefix='pre_attention')
            self.pre_attention.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1,
                                             padding=0))
            self.pre_attention.add(nn.BatchNorm())
            self.pre_attention.add(nn.Activation('relu'))

            self.spa1_attention = nn.HybridSequential(prefix='spa1_attention')
            self.spa1_attention.add(nn.Conv2D(inter_channels, kernel_size=3, strides=1,
                                              dilation=act_dilation_1, padding=act_dilation_1,
                                              groups=inter_channels))
            self.spa1_attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

            self.bn = nn.BatchNorm()
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xa = F.concat(x3, x5, dim=1)
        xa = self.attention(xa)
        xs = x3 * xa + x5 * (1 - xa)

        xs = self.bn(xs)
        xs = self.relu(xs)

        return xs
