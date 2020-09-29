from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from .askc import ResBlockV2ASKC, ResGlobLocaChaFuse, DirectAddFuse, ASKCFuse, \
    ResGlobGlobChaFuse, ResLocaLocaChaFuse

from .convolution import LearnedConv, ChaDyReFConv, SK_ChaDyReFConv, SK_1x1DepthDyReFConv, \
    SK_MSSpaDyReFConv, Direct_AddConv, SK_SpaDyReFConv, SKConv, SeqDyReFConv, \
    SK_SeqDyReFConv, SK_LGChaDyReFConv, SK_TwoLGChaDyReFConv, SK_LadderLGChaDyReFConv, \
    SK_AAPChaDyReFConv, MatryoshkaDyReFConv, \
    AYforXplusYDyReFConv, XplusAYforYDyReFConv, AXYforXplusYDyReFConv


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class ResBlockV2DyReF(HybridBlock):
    def __init__(self, conv_mode, dilations, act_dilation, channels, asBackbone,
                 stride, downsample=False, in_channels=0, norm_layer=BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(ResBlockV2DyReF, self).__init__(**kwargs)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu1 = nn.Activation('relu')
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu2 = nn.Activation('relu')

        self.conv1 = self.get_conv(conv_mode=conv_mode, channels=in_channels,
                                   dilations=dilations, asBackbone=asBackbone,
                                   act_dilation=act_dilation, stride=stride)
        self.conv2 = self.get_conv(conv_mode=conv_mode, channels=channels,
                                   dilations=dilations, asBackbone=asBackbone,
                                   act_dilation=act_dilation, stride=1)

        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def get_conv(self, conv_mode, channels, dilations, asBackbone, act_dilation, stride):

        if conv_mode == 'learned':
            return LearnedConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'ChaDyReF':
            return ChaDyReFConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_ChaDyReF':
            return SK_ChaDyReFConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_1x1DepthDyReF':
            return SK_1x1DepthDyReFConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_MSSpaDyReF':
            return SK_MSSpaDyReFConv(channels=channels, dilations=dilations,
                                        asBackbone=asBackbone, stride=stride)
        elif conv_mode == 'Direct_Add':
            return Direct_AddConv(channels=channels, dilations=dilations,
                                        asBackbone=asBackbone, stride=stride)
        elif conv_mode == 'SK_SpaDyReF':
            return SK_SpaDyReFConv(channels=channels, dilations=dilations,
                                      act_dilation=act_dilation, stride=stride)
        elif conv_mode == 'SKCell':
            return SKConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_SeqDyReF':
            return SK_SeqDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                   asBackbone=asBackbone)
        elif conv_mode == 'SK_LGChaDyReF':
            return SK_LGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_TwoLGChaDyReF':
            return SK_TwoLGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_LadderLGChaDyReF':
            return SK_LadderLGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_AAPChaDyReF':
            return SK_AAPChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'MatryoshkaDyReF':
            return MatryoshkaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'AYforXplusYDyReF':
            return AYforXplusYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'XplusAYforYDyReF':
            return XplusAYforYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'AXYforXplusYDyReF':
            return AXYforXplusYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        else:
            raise ValueError('Unknown conv_mode')

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)
        return x + residual


class ResNet20V2DyReF(HybridBlock):
    def __init__(self, layers, channels, classes, conv_mode, dilations, act_dilation,
                 asBackbone, useGlobal, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNet20V2DyReF, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(self.get_conv(conv_mode=conv_mode, channels=channels[0],
                                            dilations=dilations, asBackbone=asBackbone,
                                            useGlobal=useGlobal, act_dilation=act_dilation,
                                            stride=1))
            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(
                    layers=num_layer, channels=channels[i+1], in_channels=in_channels,
                    stride=stride, stage_index=i+1, conv_mode=conv_mode, dilations=dilations,
                    act_dilation=act_dilation, asBackbone=asBackbone, useGlobal=useGlobal,
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes, in_units=in_channels)


    def _make_layer(self, layers, channels, in_channels, stride, stage_index,
                    conv_mode, dilations, act_dilation, asBackbone, useGlobal,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(ResBlockV2DyReF(conv_mode, dilations, act_dilation, channels,
                                      asBackbone, stride, channels != in_channels,
                                      in_channels, BatchNorm))
            for bidx in range(layers-1):
                layer.add(ResBlockV2DyReF(conv_mode, dilations, act_dilation, channels,
                                          asBackbone, 1, False,
                                          in_channels, BatchNorm))
        return layer

    def get_conv(self, conv_mode, channels, dilations, asBackbone, useGlobal,
                 act_dilation, stride):

        if conv_mode == 'learned':
            return LearnedConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'ChaDyReF':
            return ChaDyReFConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_ChaDyReF':
            return SK_ChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                   useGlobal=useGlobal)
        elif conv_mode == 'SK_1x1DepthDyReF':
            return SK_1x1DepthDyReFConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SK_MSSpaDyReF':
            return SK_MSSpaDyReFConv(channels=channels, dilations=dilations,
                                        asBackbone=asBackbone, stride=stride)
        elif conv_mode == 'Direct_Add':
            return Direct_AddConv(channels=channels, dilations=dilations,
                                        asBackbone=asBackbone, stride=stride)
        elif conv_mode == 'SK_SpaDyReF':
            return SK_SpaDyReFConv(channels=channels, dilations=dilations,
                                      act_dilation=act_dilation, stride=stride)
        elif conv_mode == 'SKCell':
            return SKConv(channels=channels, dilations=dilations, stride=stride)
        elif conv_mode == 'SeqDyReF':
            return SeqDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_SeqDyReF':
            return SK_SeqDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_LGChaDyReF':
            return SK_LGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                asBackbone=asBackbone)
        elif conv_mode == 'SK_TwoLGChaDyReF':
            return SK_TwoLGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)
        elif conv_mode == 'SK_LadderLGChaDyReF':
            return SK_LadderLGChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)
        elif conv_mode == 'SK_AAPChaDyReF':
            return SK_AAPChaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)
        elif conv_mode == 'MatryoshkaDyReF':
            return MatryoshkaDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)

        elif conv_mode == 'AYforXplusYDyReF':
            return AYforXplusYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)
        elif conv_mode == 'XplusAYforYDyReF':
            return XplusAYforYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)
        elif conv_mode == 'AXYforXplusYDyReF':
            return AXYforXplusYDyReFConv(channels=channels, dilations=dilations, stride=stride,
                                    asBackbone=asBackbone)

        else:
            raise ValueError('Unknown conv_mode')


    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class ResNet20V2ASKC(HybridBlock):
    def __init__(self, layers, channels, classes, askc_type, asBackbone,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNet20V2ASKC, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(
                    layers=num_layer, channels=channels[i+1], in_channels=in_channels,
                    stride=stride, stage_index=i+1, askc_type=askc_type,
                    asBackbone=asBackbone, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, layers, channels, in_channels, stride, stage_index,
                    askc_type, asBackbone, norm_layer=BatchNorm,
                    norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(ResBlockV2ASKC(
                askc_type, channels=channels, stride=stride, downsample=channels!=in_channels,
                in_channels=in_channels, asBackbone=asBackbone,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for bidx in range(layers-1):
                layer.add(ResBlockV2ASKC(
                    askc_type, channels=channels, stride=1, downsample=False,
                    in_channels=channels, asBackbone=asBackbone,
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class ResNet50_v1bASKC(HybridBlock):
    def __init__(self, askc_type, layers, classes=1000, dilated=False, norm_layer=BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNet50_v1bASKC, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                   padding=3, use_bias=False)
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.layer1 = self._make_layer('DirectAdd', 1, BottleneckV1bASKC, 64, layers[0],
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
            self.layer2 = self._make_layer('DirectAdd', 2, BottleneckV1bASKC, 128, layers[1],
                                           strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(askc_type, 3, BottleneckV1bASKC, 256, layers[2],
                                               strides=1, dilation=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                self.layer4 = self._make_layer(askc_type, 4, BottleneckV1bASKC, 512, layers[3],
                                               strides=1, dilation=4, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(askc_type, 3, BottleneckV1bASKC, 256, layers[2],
                                               strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                self.layer4 = self._make_layer(askc_type, 4, BottleneckV1bASKC, 512, layers[3],
                                               strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)

            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * BottleneckV1bASKC.expansion, units=classes)

    def _make_layer(self, askc_type, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                if avg_down:
                    if dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(BottleneckV1bASKC(
                    askc_type, planes, strides, dilation=1, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            elif dilation == 4:
                layers.add(BottleneckV1bASKC(
                    askc_type, planes, strides, dilation=2, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(BottleneckV1bASKC(
                    askc_type, planes, dilation=dilation, previous_dilation=dilation,
                    norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                    last_gamma=last_gamma))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


class BottleneckV1bASKC(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, askc_type, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(BottleneckV1bASKC, self).__init__()
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

        if askc_type == 'ASKCFuse':
            self.attention = ASKCFuse(channels=planes*4, r=16)
        elif askc_type == 'ResGlobGlobCha':
            self.attention = ResGlobGlobChaFuse(channels=planes*4, r=16)
        elif askc_type == 'ResLocaLocaCha':
            self.attention = ResLocaLocaChaFuse(channels=planes*4, r=16)
        elif askc_type == 'DirectAdd':
            self.attention = DirectAddFuse(channels=planes*4)
        else:
            raise ValueError('unknown askc type')

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
