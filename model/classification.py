from __future__ import division

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from .block import AFFResBlockV2, AFFBottleneck, AFFResNeXtBlock


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class CIFARAFFResNet(HybridBlock):
    def __init__(self, askc_type, start_layer, layers, channels, classes, deep_stem,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(CIFARAFFResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            # self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            stem_width = channels[0] // 2
            if not deep_stem:
                self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            else:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(
                    askc_type=askc_type, start_layer=start_layer, layers=num_layer,
                    channels=channels[i+1], in_channels=in_channels, stride=stride,
                    stage_index=i+1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, askc_type, start_layer, layers, channels, in_channels, stride,
                    stage_index, norm_layer=BatchNorm, norm_kwargs=None):
        if stage_index < start_layer:
            askc_type = 'DirectAdd'
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(AFFResBlockV2(
                askc_type=askc_type, channels=channels, stride=stride,
                downsample=channels != in_channels, in_channels=in_channels,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for bidx in range(layers-1):
                layer.add(AFFResBlockV2(
                    askc_type=askc_type, channels=channels, stride=1, downsample=False,
                    in_channels=channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class CIFARAFFResNeXt(HybridBlock):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, askc_type, start_layer, layers, channels, cardinality, bottleneck_width,
                 classes=10, deep_stem=False, use_se=False, norm_layer=BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(CIFARAFFResNeXt, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = channels[0]
        stem_width = channels//2

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if not deep_stem:
                self.features.add(nn.Conv2D(channels, 3, 1, 1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
            else:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(
                    askc_type, start_layer, use_se, channels, num_layer, stride, i+1,
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                channels *= 2
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, askc_type, start_layer, use_se, channels, num_layer, stride, stage_index,
                    norm_layer=BatchNorm, norm_kwargs=None):
        if stage_index < start_layer:
            askc_type = 'DirectAdd'
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(AFFResNeXtBlock(
                askc_type, channels, self.cardinality, self.bottleneck_width,
                stride, True, use_se=use_se, prefix='',
                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(num_layer-1):
                layer.add(AFFResNeXtBlock(
                    askc_type, channels, self.cardinality, self.bottleneck_width, 1, False,
                    use_se=use_se, prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


class AFFResNet(HybridBlock):
    def __init__(self, askc_type, start_layer, layers, classes=1000, dilated=False,
                 norm_layer=BatchNorm, norm_kwargs=None, last_gamma=False, deep_stem=True,
                 stem_width=32, avg_down=True, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(AFFResNet, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs

        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.layer1 = self._make_layer(askc_type, start_layer, 1, AFFBottleneck, 64,
                                           layers[0], avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
            self.layer2 = self._make_layer(askc_type, start_layer, 2, AFFBottleneck, 128,
                                           layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(askc_type, start_layer, 3, AFFBottleneck, 256,
                                               layers[2], strides=1, dilation=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                self.layer4 = self._make_layer(askc_type, start_layer, 4, AFFBottleneck, 512,
                                               layers[3], strides=1, dilation=4, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(askc_type, start_layer, 3, AFFBottleneck, 256,
                                               layers[2], strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                self.layer4 = self._make_layer(askc_type, start_layer, 4, AFFBottleneck, 512,
                                               layers[3], strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)

            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * AFFBottleneck.expansion, units=classes)

    def _make_layer(self, askc_type, start_layer, stage_index, block, planes, blocks, strides=1,
                    dilation=1, avg_down=False, norm_layer=None, last_gamma=False):
        if stage_index < start_layer:
            askc_type = 'DirectAdd'
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
                layers.add(AFFBottleneck(
                    askc_type, planes, strides, dilation=1, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            elif dilation == 4:
                layers.add(AFFBottleneck(
                    askc_type, planes, strides, dilation=2, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(AFFBottleneck(
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


class AFFResNeXt(HybridBlock):
    r"""ResNext model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 1000
        Number of classification classes.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    stem_width : int, default 64
        Width of the stem intermediate layer.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, askc_type, start_layer, layers, cardinality, bottleneck_width,
                 classes=1000, last_gamma=False, use_se=False, deep_stem=True, avg_down=True,
                 stem_width=64, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(AFFResNeXt, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if not deep_stem:
                self.features.add(nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                            padding=3, use_bias=False))
            else:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width * 2, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(
                    askc_type, start_layer, channels, num_layer, stride, last_gamma, use_se,
                    False if i == 0 else avg_down, i + 1,
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                channels *= 2
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, askc_type, start_layer, channels, num_layers, stride, last_gamma,
                    use_se, avg_down, stage_index, norm_layer=BatchNorm, norm_kwargs=None):
        if stage_index < start_layer:
            askc_type = 'DirectAdd'
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(AFFResNeXtBlock(
                askc_type, channels, self.cardinality, self.bottleneck_width, stride, True,
                last_gamma=last_gamma, use_se=use_se, avg_down=avg_down,
                prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(num_layers - 1):
                layer.add(AFFResNeXtBlock(
                    askc_type, channels, self.cardinality, self.bottleneck_width,
                    1, False, last_gamma=last_gamma, use_se=use_se, prefix='',
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x
