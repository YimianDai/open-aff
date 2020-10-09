from __future__ import division

import math
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from .fusion import DirectAddFuse, ASKCFuse, ResGlobLocaforGlobLocaChaFuse


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class AFFResBlockV2(HybridBlock):
    def __init__(self, askc_type, channels, stride, downsample=False, in_channels=0,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(AFFResBlockV2, self).__init__(**kwargs)

        self.resfwd = nn.HybridSequential(prefix='resfwd')
        self.resfwd.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.resfwd.add(nn.Activation('relu'))
        self.resfwd.add(_conv3x3(channels, stride, in_channels))
        self.resfwd.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.resfwd.add(nn.Activation('relu'))
        self.resfwd.add(_conv3x3(channels, 1, channels))

        self.downsample = nn.HybridSequential(prefix='downsample')
        if downsample:
            self.downsample.add(nn.AvgPool2D(pool_size=stride, strides=stride,
                                             ceil_mode=True, count_include_pad=False))
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                                          use_bias=False, in_channels=in_channels))

        if askc_type == 'DirectAdd':
            self.attention = DirectAddFuse()
        elif askc_type == 'ResGlobLocaforGlobLocaCha':
            self.attention = ResGlobLocaforGlobLocaChaFuse(channels=channels)
        elif askc_type == 'ASKCFuse':
            self.attention = ASKCFuse(channels=channels)
        else:
            raise ValueError('Unknown askc_type')

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.resfwd(x)

        if self.downsample:
            residual = self.downsample(residual)

        xo = self.attention(x, residual)

        return xo


class AFFBottleneck(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, askc_type, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(AFFBottleneck, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=False)
        self.bn1 = norm_layer(in_channels=planes, **norm_kwargs)
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=False)
        self.bn2 = norm_layer(in_channels=planes, **norm_kwargs)

        self.relu1 = nn.Activation('relu')
        self.relu2 = nn.Activation('relu')
        self.relu3 = nn.Activation('relu')

        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(in_channels=planes*4, **norm_kwargs)
        else:
            self.bn3 = norm_layer(in_channels=planes*4, gamma_initializer='zeros',
                                  **norm_kwargs)

        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

        if askc_type == 'DirectAdd':
            self.attention = DirectAddFuse()
        elif askc_type == 'ResGlobLocaforGlobLocaCha':
            self.attention = ResGlobLocaforGlobLocaChaFuse(channels=planes*4, r=16)
        elif askc_type == 'ASKCFuse':
            self.attention = ASKCFuse(channels=planes*4, r=16)
        else:
            raise ValueError('Unknown askc_type')

    def hybrid_forward(self, F, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.attention(out, residual)
        out = self.relu3(out)

        return out


class AFFResNeXtBlock(HybridBlock):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, askc_type, channels, cardinality, bottleneck_width, stride,
                 downsample=False, last_gamma=False, use_se=False, avg_down=True,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(AFFResNeXtBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width, kernel_size=1, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1,
                                groups=cardinality, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, use_bias=False))
        if last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Conv2D(channels // 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Conv2D(channels * 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            if avg_down:
                self.downsample.add(nn.AvgPool2D(pool_size=stride, strides=stride,
                                                 ceil_mode=True, count_include_pad=False))
                self.downsample.add(nn.Conv2D(channels=channels * 4, kernel_size=1,
                                              strides=1, use_bias=False))
            else:
                self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                              use_bias=False))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

        if askc_type == 'DirectAdd':
            self.attention = DirectAddFuse()
        elif askc_type == 'ResGlobLocaforGlobLocaCha':
            self.attention = ResGlobLocaforGlobLocaChaFuse(channels=channels*4, r=16)
        elif askc_type == 'ASKCFuse':
            self.attention = ASKCFuse(channels=channels*4, r=16)
        else:
            raise ValueError('Unknown askc_type')

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w)

        if self.downsample:
            residual = self.downsample(residual)

        out = self.attention(x, residual)

        x = F.Activation(out, act_type='relu')
        return x
