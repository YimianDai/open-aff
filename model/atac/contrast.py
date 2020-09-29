from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from gluoncv.model_zoo.fcn import _FCNHead
from mxnet import nd


class PCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, shift='xxx', **kwargs):
        super(PCMNet, self).__init__(**kwargs)

        self.shift = shift
        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)
        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=self.shift)
        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)

        c12 = nd.minimum(s1, s2)
        c123 = nd.minimum(c12, s3)
        c1234 = nd.minimum(c123, s4)
        x = self.head(c1234)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)

    def circ_shift(self, cen, shift):

        _, _, hei, wid = cen.shape

        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
        B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
        B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
        B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
        B1_N = nd.concat(B1_NW, B1_NE, dim=3)
        B1_S = nd.concat(B1_SW, B1_SE, dim=3)
        B1 = nd.concat(B1_N, B1_S, dim=2)

        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
        B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
        B2 = nd.concat(B2_N, B2_S, dim=2)

        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
        B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
        B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
        B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
        B3_N = nd.concat(B3_NW, B3_NE, dim=3)
        B3_S = nd.concat(B3_SW, B3_SE, dim=3)
        B3 = nd.concat(B3_N, B3_S, dim=2)

        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
        B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
        B4 = nd.concat(B4_W, B4_E, dim=3)

        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
        B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
        B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
        B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
        B5_N = nd.concat(B5_NW, B5_NE, dim=3)
        B5_S = nd.concat(B5_SW, B5_SE, dim=3)
        B5 = nd.concat(B5_N, B5_S, dim=2)

        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
        B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
        B6 = nd.concat(B6_N, B6_S, dim=2)

        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
        B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
        B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
        B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
        B7_N = nd.concat(B7_NW, B7_NE, dim=3)
        B7_S = nd.concat(B7_SW, B7_SE, dim=3)
        B7 = nd.concat(B7_N, B7_S, dim=2)

        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
        B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
        B8 = nd.concat(B8_W, B8_E, dim=3)

        return B1, B2, B3, B4, B5, B6, B7, B8


class MPCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(MPCMNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)

        # pcm9 = self.cal_pcm(cen, shift=9)
        # pcm17 = self.cal_pcm(cen, shift=17)
        # pcm25 = self.cal_pcm(cen, shift=25)
        # pcm33 = self.cal_pcm(cen, shift=33)
        # mpcm = nd.maximum(nd.maximum(nd.maximum(pcm9, pcm17), pcm25), pcm33)

        pcm9 = self.cal_pcm(cen, shift=9)
        pcm13 = self.cal_pcm(cen, shift=13)
        pcm17 = self.cal_pcm(cen, shift=17)
        # pcm21 = self.cal_pcm(cen, shift=21)
        # mpcm = nd.maximum(nd.maximum(nd.maximum(pcm9, pcm13), pcm17), pcm21)
        mpcm = nd.maximum(nd.maximum(pcm9, pcm13), pcm17)


        x = self.head(mpcm)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)

    def circ_shift(self, cen, shift):

        _, _, hei, wid = cen.shape

        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
        B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
        B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
        B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
        B1_N = nd.concat(B1_NW, B1_NE, dim=3)
        B1_S = nd.concat(B1_SW, B1_SE, dim=3)
        B1 = nd.concat(B1_N, B1_S, dim=2)

        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
        B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
        B2 = nd.concat(B2_N, B2_S, dim=2)

        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
        B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
        B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
        B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
        B3_N = nd.concat(B3_NW, B3_NE, dim=3)
        B3_S = nd.concat(B3_SW, B3_SE, dim=3)
        B3 = nd.concat(B3_N, B3_S, dim=2)

        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
        B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
        B4 = nd.concat(B4_W, B4_E, dim=3)

        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
        B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
        B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
        B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
        B5_N = nd.concat(B5_NW, B5_NE, dim=3)
        B5_S = nd.concat(B5_SW, B5_SE, dim=3)
        B5 = nd.concat(B5_N, B5_S, dim=2)

        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
        B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
        B6 = nd.concat(B6_N, B6_S, dim=2)

        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
        B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
        B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
        B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
        B7_N = nd.concat(B7_NW, B7_NE, dim=3)
        B7_S = nd.concat(B7_SW, B7_SE, dim=3)
        B7 = nd.concat(B7_N, B7_S, dim=2)

        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
        B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
        B8 = nd.concat(B8_W, B8_E, dim=3)

        return B1, B2, B3, B4, B5, B6, B7, B8

    def cal_pcm(self, cen, shift):

        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
        # s1 = nd.relu(B1 - cen) * nd.relu(B5 - cen)
        # s2 = nd.relu(B2 - cen) * nd.relu(B6 - cen)
        # s3 = nd.relu(B3 - cen) * nd.relu(B7 - cen)
        # s4 = nd.relu(B4 - cen) * nd.relu(B8 - cen)

        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)  

        c12 = nd.minimum(s1, s2)
        c123 = nd.minimum(c12, s3)
        c1234 = nd.minimum(c123, s4)

        return c1234


class LayerwiseMPCMNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(LayerwiseMPCMNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(CalMPCM())

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))
            layer.add(CalMPCM())

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        x = self.features(x)
        x = self.head(x)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class PlainNet(HybridBlock):
    def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1, addstem=True,
                 maxpool=False, **kwargs):
        super(PlainNet, self).__init__(**kwargs)

        stem_width = int(channels // 2)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if addstem:
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                            padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(in_channels=stem_width*2))
                self.features.add(nn.Activation('relu'))
            if maxpool:
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, dilation in enumerate(dilations):
                self.features.add(self._make_layer(
                    dilation=dilation, channels=channels, stage_index=i))

            self.head = _FCNHead(in_channels=channels, channels=classes)

    def _make_layer(self, dilation, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():

            layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
                                padding=dilation))
            layer.add(nn.BatchNorm())
            layer.add(nn.Activation('relu'))

        return layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        cen = self.features(x)

        x = self.head(cen)

        out = F.contrib.BilinearResize2D(x, height=hei, width=wid)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)



class CalMPCM(HybridBlock):
    def __init__(self, **kwargs):
        super(CalMPCM, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):

        pcm9 = self.cal_pcm(x, shift=9)
        pcm13 = self.cal_pcm(x, shift=13)
        pcm17 = self.cal_pcm(x, shift=17)
        mpcm = nd.maximum(nd.maximum(pcm9, pcm13), pcm17)

        return mpcm

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)

    def circ_shift(self, cen, shift):

        _, _, hei, wid = cen.shape

        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
        B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
        B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
        B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
        B1_N = nd.concat(B1_NW, B1_NE, dim=3)
        B1_S = nd.concat(B1_SW, B1_SE, dim=3)
        B1 = nd.concat(B1_N, B1_S, dim=2)

        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
        B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
        B2 = nd.concat(B2_N, B2_S, dim=2)

        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
        B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
        B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
        B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
        B3_N = nd.concat(B3_NW, B3_NE, dim=3)
        B3_S = nd.concat(B3_SW, B3_SE, dim=3)
        B3 = nd.concat(B3_N, B3_S, dim=2)

        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
        B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
        B4 = nd.concat(B4_W, B4_E, dim=3)

        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
        B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
        B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
        B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
        B5_N = nd.concat(B5_NW, B5_NE, dim=3)
        B5_S = nd.concat(B5_SW, B5_SE, dim=3)
        B5 = nd.concat(B5_N, B5_S, dim=2)

        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
        B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
        B6 = nd.concat(B6_N, B6_S, dim=2)

        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
        B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
        B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
        B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
        B7_N = nd.concat(B7_NW, B7_NE, dim=3)
        B7_S = nd.concat(B7_SW, B7_SE, dim=3)
        B7 = nd.concat(B7_N, B7_S, dim=2)

        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
        B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
        B8 = nd.concat(B8_W, B8_E, dim=3)

        return B1, B2, B3, B4, B5, B6, B7, B8

    def cal_pcm(self, cen, shift):

        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)

        c12 = nd.minimum(s1, s2)
        c123 = nd.minimum(c12, s3)
        c1234 = nd.minimum(c123, s4)

        return c1234
