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




def _conv3x3(channels, stride, dilation=1):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=dilation,
                     dilation=dilation, use_bias=False)


class ATACBlockV1(HybridBlock):
    def __init__(self, channels, stride, useReLU, act_type, skernel, dilation,
                 downsample=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ATACBlockV1, self).__init__(**kwargs)
        assert act_type in ['relu', 'prelu', 'xUnit', 'ChaATAC', 'SpaATAC',
                            'SeqATAC'], "Unknown act_type in ATACBlockV2"
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(channels, stride)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, 1)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False)
        else:
            self.downsample = None

        if act_type == 'relu':
            self.act1 = nn.Activation('relu')
            self.act2 = nn.Activation('relu')
        elif act_type == 'prelu':
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
        elif act_type == 'xUnit':
            # self.act1 = xUnit(channels=channels, skernel_size=skernel)
            # self.act2 = xUnit(channels=channels, skernel_size=skernel)
            self.act1 = xUnit(channels=channels, skernel_size=9)
            self.act2 = xUnit(channels=channels, skernel_size=9)
        elif act_type == 'SpaATAC':
            self.act1 = SpaATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU, asBackbone=False)
            self.act2 = SpaATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU, asBackbone=False)
        elif act_type == 'ChaATAC':
            self.act1 = ChaATAC(channels=channels, useReLU=useReLU, useGlobal=False,
                                asBackbone=False)
            self.act2 = ChaATAC(channels=channels, useReLU=useReLU, useGlobal=False,
                                asBackbone=False)
        elif act_type == 'SeqATAC':
            self.act1 = SeqATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU, asBackbone=False)
            self.act2 = SeqATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU, asBackbone=False)
        else:
            raise ValueError('Unknown act_type')

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.act2(out)

        return out


class conv1ATAC(HybridBlock):
    def __init__(self, channels, stride, useReLU, act_type, skernel, dilation,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(conv1ATAC, self).__init__(**kwargs)
        assert act_type in ['relu', 'prelu', 'xUnit', 'ChaATAC', 'SpaATAC',
                            'SeqATAC'], "Unknown act_type in ATACBlockV2"
        self.conv1 = _conv3x3(channels, stride)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        if act_type == 'relu':
            self.act1 = nn.Activation('relu')
        elif act_type == 'prelu':
            self.act1 = nn.PReLU()
        elif act_type == 'xUnit':
            self.act1 = xUnit(channels=channels, skernel_size=skernel)
        elif act_type == 'SpaATAC':
            self.act1 = SpaATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU)
        elif act_type == 'ChaATAC':
            self.act1 = ChaATAC(channels=channels, useReLU=useReLU)
        elif act_type == 'SeqATAC':
            self.act1 = SeqATAC(skernel=skernel, channels=channels, dilation=dilation,
                                useReLU=useReLU)
        else:
            raise ValueError('Unknown act_type')

    def hybrid_forward(self, F, x):
        """Hybrid forward"""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x



class DynamicCell(HybridBlock):
    def __init__(self, channels=64, dilations=(1, 2)):
        super(DynamicCell, self).__init__()
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

            self.act = MSSeqATACConcat(skernel=-1, channels=channels, dilation=(4, 8),
                                       useReLU=False, asBackbone=False)

    def hybrid_forward(self, F, x):

        x3 = self.feature_spatial_3(x)
        x5 = self.feature_spatial_5(x)

        xc = F.concat(x3, x5, dim=1)
        xc = self.feature_channel(xc)
        wei = self.act(xc)

        xs = x3 * wei + x5 * (1 - wei)

        return xs
