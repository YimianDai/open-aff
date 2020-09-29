from __future__ import division
import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn



class Direct_AddFuse_Reduce(HybridBlock):
    def __init__(self, channels=64):
        super(Direct_AddFuse_Reduce, self).__init__()
        self.channels = channels

        self.feature_high = nn.HybridSequential(prefix='feature_high')
        self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                        dilation=1))
        self.feature_high.add(nn.BatchNorm())
        self.feature_high.add((nn.Activation('relu')))

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        xs = xh + xl
        xs = self.post(xs)

        return xs



class ConcatFuse_Reduce(HybridBlock):
    def __init__(self, channels=64):
        super(ConcatFuse_Reduce, self).__init__()
        self.channels = channels

        self.feature_high = nn.HybridSequential(prefix='feature_high')
        self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                        dilation=1))
        self.feature_high.add(nn.BatchNorm())
        self.feature_high.add((nn.Activation('relu')))

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        xs = F.concat(xh, xl, dim=1)
        xs = self.post(xs)

        return xs



class SKFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(SKFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)
        self.softmax_channels = int(channels * 2)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_low')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl
        xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        xa = F.softmax(xa, axis=1)

        xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        xa3 = F.reshape(xa3, (0, -1, 1, 1))
        xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs

class GlobalChaFuse(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(GlobalChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 2)
        self.softmax_channels = int(channels * 1)

        with self.name_scope():

            self.feature_low = nn.HybridSequential(prefix='feature_low')
            self.feature_low.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_low.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            # self.attention.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
            #                              padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, xh, xl):

        xl = self.feature_low(xl)

        xa = xh + xl
        # xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        # xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        # xa = F.softmax(xa, axis=1)

        # xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        # xa3 = F.reshape(xa3, (0, -1, 1, 1))
        # xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        # xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xa3 = self.attention(xa)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)

        return xs


class LocalChaFuse(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(LocalChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 2)
        self.softmax_channels = int(channels * 1)

        with self.name_scope():

            self.feature_low = nn.HybridSequential(prefix='feature_low')
            self.feature_low.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_low.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, xh, xl):

        xl = self.feature_low(xl)

        xa = xh + xl
        _, _, hei, wid = xa.shape
        inter_hei, inter_wid = int(hei // 8), int(wid // 8)
        inter_xa = F.contrib.AdaptiveAvgPooling2D(xa, (inter_hei, inter_wid))
        inter_xa3 = self.attention(inter_xa)
        xa3 = F.contrib.AdaptiveAvgPooling2D(inter_xa3, (hei, wid))

        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)

        return xs


class LocalSpaFuse(HybridBlock):
    def __init__(self, channels=64, act_dilation=-1):
        super(LocalSpaFuse, self).__init__()
        # self.channels = channels
        # self.bottleneck_channels = int(channels // 2)
        # self.softmax_channels = int(channels * 1)

        with self.name_scope():

            self.feature_low = nn.HybridSequential(prefix='feature_low')
            self.feature_low.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_low.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                         padding=act_dilation, dilation=act_dilation,
                                         groups=channels))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, xh, xl):

        xl = self.feature_low(xl)

        xa = xh + xl
        xa3 = self.attention(xa)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)

        return xs


class GlobalSpaFuse(HybridBlock):
    def __init__(self, channels=64, act_dilation=-1):
        super(GlobalSpaFuse, self).__init__()
        # self.channels = channels
        # self.bottleneck_channels = int(channels // 2)
        # self.softmax_channels = int(channels * 1)

        with self.name_scope():

            self.feature_low = nn.HybridSequential(prefix='feature_low')
            self.feature_low.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_low.add(nn.BatchNorm())

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(1, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(1, kernel_size=3, strides=1,
                                         padding=act_dilation, dilation=act_dilation))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, xh, xl):

        xl = self.feature_low(xl)

        xa = xh + xl
        xa3 = self.attention(xa)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)

        return xs

class SK_MSSpaFuse(HybridBlock):
    def __init__(self, channels=64, act_dilation=(8, 16), stride=1):
        super(SK_MSSpaFuse, self).__init__()
        self.channels = channels

        with self.name_scope():

            act_dilation_1 = act_dilation[0]
            act_dilation_2 = act_dilation[1]

            self.feature_low = nn.HybridSequential(prefix='feature_low')
            self.feature_low.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_low.add(nn.BatchNorm())

            # self.attention_4 = nn.HybridSequential(prefix='attention_4')
            # self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
            #                                dilation=4, groups=channels))
            # self.attention_4.add(nn.BatchNorm())

            self.attention_8 = nn.HybridSequential(prefix='attention_8')
            self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                           padding=act_dilation_1, dilation=act_dilation_1,
                                           groups=channels))
            self.attention_8.add(nn.BatchNorm())

            self.attention_16 = nn.HybridSequential(prefix='attention_16')
            self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                            padding=act_dilation_2, dilation=act_dilation_2,
                                            groups=channels))
            self.attention_16.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            # self.bn = nn.BatchNorm()
            # self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, xh, xl):

        xl = self.feature_low(xl)
        xa = xh + xl

        # xa4 = self.attention_4(xa)
        # xa8 = self.attention_8(xa)
        # xa16 = self.attention_16(xa)
        # xa = xa4 + xa8 + xa16

        xa8 = self.attention_8(xa)
        xa16 = self.attention_16(xa)
        xa = xa8 + xa16

        xa = self.sigmoid(xa)
        xs = 2 * xh * xa + 2 * xl * (1 - xa)

        return xs



class LocalGlobalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(LocalGlobalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)


        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.global_att.add(nn.BatchNorm())
            # self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
            #                               padding=0))
            # self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())
            # self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
            #                              padding=0))
            # self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl

        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs



class SpaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, act_dialtion="xxx"):
        super(SpaFuse_Reduce, self).__init__()
        self.channels = channels

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.spa_att = nn.HybridSequential(prefix='spa_att')
            self.spa_att.add(nn.Conv2D(self.channels, kernel_size=3, strides=1,
                                       dilation=act_dialtion, padding=act_dialtion,
                                       groups=self.channels))
            self.spa_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl

        att = self.spa_att(xa)
        xa3 = self.sigmoid(att)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs



class LocalLocalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(LocalLocalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)


        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl

        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs



class BiLocalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(BiLocalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class BiGlobalLocalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(BiGlobalLocalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown_global = nn.HybridSequential(prefix='topdown_global')
            self.topdown_global.add(nn.GlobalAvgPool2D())
            self.topdown_global.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown_global.add(nn.BatchNorm())
            self.topdown_global.add(nn.Activation('relu'))
            self.topdown_global.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown_global.add(nn.BatchNorm())

            self.topdown_local = nn.HybridSequential(prefix='topdown_local')
            self.topdown_local.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown_local.add(nn.BatchNorm())
            self.topdown_local.add(nn.Activation('relu'))
            self.topdown_local.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown_local.add(nn.BatchNorm())


            self.bottomup_local = nn.HybridSequential(prefix='bottomup_local')
            self.bottomup_local.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup_local.add(nn.BatchNorm())
            self.bottomup_local.add(nn.Activation('relu'))
            self.bottomup_local.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup_local.add(nn.BatchNorm())

            self.bottomup_global = nn.HybridSequential(prefix='bottomup_global')
            self.bottomup_global.add(nn.GlobalAvgPool2D())
            self.bottomup_global.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup_global.add(nn.BatchNorm())
            self.bottomup_global.add(nn.Activation('relu'))
            self.bottomup_global.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup_global.add(nn.BatchNorm())

            self.bottomup_sig = nn.Activation('sigmoid')
            self.topdown_sig = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei_global = self.topdown_global(xh)
        topdown_wei_local = self.topdown_local(xh)
        topdown_wei = self.topdown_sig(topdown_wei_global + topdown_wei_local)

        bottomup_wei_global = self.bottomup_global(xl)
        bottomup_wei_local = self.bottomup_local(xl)
        bottomup_wei = self.bottomup_sig(bottomup_wei_global + bottomup_wei_local)

        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class AsymBiLocalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(AsymBiLocalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs



class BiSpaChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(BiSpaChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown_cha = nn.HybridSequential(prefix='topdown_cha')
            self.topdown_cha.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown_cha.add(nn.BatchNorm())
            self.topdown_cha.add(nn.Activation('relu'))
            self.topdown_cha.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown_cha.add(nn.BatchNorm())

            self.topdown_spa = nn.HybridSequential(prefix='topdown_spa')
            self.topdown_spa.add(nn.Conv2D(self.channels, kernel_size=3, strides=1,
                                          padding=1, groups=self.channels))
            self.topdown_spa.add(nn.BatchNorm())

            self.bottomup_cha = nn.HybridSequential(prefix='bottomup_cha')
            self.bottomup_cha.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup_cha.add(nn.BatchNorm())
            self.bottomup_cha.add(nn.Activation('relu'))
            self.bottomup_cha.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup_cha.add(nn.BatchNorm())

            self.bottomup_spa = nn.HybridSequential(prefix='topdown_spa')
            self.bottomup_spa.add(nn.Conv2D(self.channels, kernel_size=3, strides=1,
                                          padding=1, groups=self.channels))
            self.bottomup_spa.add(nn.BatchNorm())

            self.bottomup_sig = nn.Activation('sigmoid')
            self.topdown_sig = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        topdown_spa_wei = self.topdown_spa(xh)
        topdown_cha_wei = self.topdown_cha(xh)
        topdown_wei = self.topdown_sig(topdown_spa_wei + topdown_cha_wei)

        bottomup_spa_wei = self.bottomup_spa(xl)
        bottomup_cha_wei = self.bottomup_cha(xl)
        bottomup_wei = self.bottomup_sig(bottomup_spa_wei + bottomup_cha_wei)

        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs



class AsymBiSpaChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(AsymBiSpaChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown_cha')
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=3, strides=1,
                                          padding=1, groups=self.channels))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))


            self.bottomup = nn.HybridSequential(prefix='bottomup_cha')
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=3, strides=1,
                                          padding=1, groups=self.channels))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)

        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs



class BiGlobalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(BiGlobalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.GlobalAvgPool2D())
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs



class GlobalGlobalChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(GlobalGlobalChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)


        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl

        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)
        xa5 = 1 - xa3
        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs


class AYforXplusYChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(AYforXplusYChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh
        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs


class AXYforXplusYChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(AXYforXplusYChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl
        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs



class GAUChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(GAUChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs


class LocalGAUChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(LocalGAUChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            # self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs



class XplusAYforYChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64, stride=1):
        super(XplusAYforYChaFuse_Reduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.local_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl
        al = self.local_att(xa)
        ag = self.global_att(xa)
        att = F.broadcast_add(al, ag)
        xa3 = self.sigmoid(att)

        xs = F.broadcast_mul(xh, xa3) + xl
        xs = self.post(xs)

        return xs


class IASKCChaFuse_Reduce(HybridBlock):
    def __init__(self, channels=64):
        super(IASKCChaFuse_Reduce, self).__init__()
        inter_channels = int(channels // 4)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.local_att2 = nn.HybridSequential(prefix='local_att2')
            self.local_att2.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att2.add(nn.BatchNorm())
            self.local_att2.add(nn.Activation('relu'))
            self.local_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att2.add(nn.BatchNorm())

            self.global_att2 = nn.HybridSequential(prefix='global_att2')
            self.global_att2.add(nn.GlobalAvgPool2D())
            self.global_att2.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att2.add(nn.BatchNorm())
            self.global_att2.add(nn.Activation('relu'))
            self.global_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att2.add(nn.BatchNorm())

            self.sig1 = nn.Activation('sigmoid')
            self.sig2 = nn.Activation('sigmoid')
            # self.sig1 = nn.Activation('relu')
            # self.sig2 = nn.Activation('relu')
            
            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))            

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig1(xlg)

        xi = 2 * F.broadcast_mul(xh, wei) + 2 * F.broadcast_mul(xl, 1-wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = F.broadcast_add(xl2, xg2)
        wei2 = self.sig2(xlg2)
        xo = 2 * F.broadcast_mul(xh, wei2) + 2 * F.broadcast_mul(xl, 1-wei2)
        
        xo = self.post(xo)

        return xo


#
# class SK_ChaDyReFConv(HybridBlock):
#
#     def __init__(self, channels=64, dilations=(1, 2), stride=1, useGlobal=False):
#         super(SK_ChaDyReFConv, self).__init__()
#         self.channels = channels
#
#         with self.name_scope():
#
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             if useGlobal:
#                 self.attention.add(nn.GlobalAvgPool2D())
#             self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             # self.attention.add(nn.BatchNorm())
#             # self.attention.add(nn.Activation('relu'))
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             self.attention.add(nn.BatchNorm())
#             self.attention.add(nn.Activation('sigmoid'))
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5
#         xa = self.attention(xa)
#         # xs = x3 * xa + x5 * (1 - xa)
#         xs = F.broadcast_mul(x3, xa) + F.broadcast_mul(x5, 1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
#
#
# class SK_SpaDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, asBackbone=False):
#         super(SK_SpaDyReFConv, self).__init__()
#         self.channels = channels
#         self.asBackbone = asBackbone
#
#         with self.name_scope():
#
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             if self.asBackbone:
#                 self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                              padding=act_dilation, dilation=act_dilation,
#                                              groups=channels))
#             else:
#                 self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                              padding=act_dilation, dilation=act_dilation,
#                                              groups=channels))
#                 self.attention.add(nn.BatchNorm())
#                 self.attention.add(nn.Activation('sigmoid'))
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5
#         if self.asBackbone:
#             xs = self.attention(xa)
#         else:
#             xa = self.attention(xa)
#             xs = x3 * xa + x5 * (1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
#
#
# class SK_1x1DepthDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), stride=1):
#         super(SK_1x1DepthDyReFConv, self).__init__()
#         self.channels = channels
#
#         with self.name_scope():
#
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
#                                          groups=channels))
#             self.attention.add(nn.BatchNorm())
#             self.attention.add(nn.Activation('relu'))
#
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
#             #                              groups=channels))
#             # self.attention.add(nn.BatchNorm())
#             # self.attention.add(nn.Activation('relu'))
#
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
#             #                              groups=channels))
#             # self.attention.add(nn.BatchNorm())
#             # self.attention.add(nn.Activation('relu'))
#
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
#             #                              groups=channels))
#             # self.attention.add(nn.BatchNorm())
#             # self.attention.add(nn.Activation('relu'))
#
#             self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
#                                          groups=channels))
#             self.attention.add(nn.BatchNorm())
#             self.attention.add(nn.Activation('sigmoid'))
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5
#         xa = self.attention(xa)
#         xs = x3 * xa + x5 * (1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
#
#
#
# class SK_MSSpaDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), asBackbone=False, stride=1):
#         super(SK_MSSpaDyReFConv, self).__init__()
#         self.channels = channels
#         self.asBackbone = asBackbone
#
#         with self.name_scope():
#
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention_4 = nn.HybridSequential(prefix='attention_4')
#             self.attention_4.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=4,
#                                            dilation=4, groups=channels))
#             self.attention_4.add(nn.BatchNorm())
#
#             self.attention_8 = nn.HybridSequential(prefix='attention_8')
#             self.attention_8.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=8,
#                                            dilation=8, groups=channels))
#             self.attention_8.add(nn.BatchNorm())
#
#             self.attention_16 = nn.HybridSequential(prefix='attention_16')
#             self.attention_16.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=16,
#                                            dilation=16, groups=channels))
#             self.attention_16.add(nn.BatchNorm())
#
#             self.sigmoid = nn.Activation('sigmoid')
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5
#         xa4 = self.attention_4(xa)
#         xa8 = self.attention_8(xa)
#         xa16 = self.attention_16(xa)
#         xa = xa4 + xa8 + xa16
#         # xa = xa8 + xa16
#
#         if self.asBackbone:
#             xs = xa
#         else:
#             xa = self.sigmoid(xa)
#             xs = x3 * xa + x5 * (1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
#
#
# class SK_SpaDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), act_dilation=1, stride=1):
#         super(SK_SpaDyReFConv, self).__init__()
#         self.channels = channels
#
#         with self.name_scope():
#
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                          padding=act_dilation, dilation=act_dilation,
#                                          groups=channels))
#             self.attention.add(nn.BatchNorm())
#
#             self.sigmoid = nn.Activation('sigmoid')
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5
#         xa = self.attention(xa)
#         xa = self.sigmoid(xa)
#
#         xs = x3 * xa + x5 * (1 - xa)
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
#
#
# class SeqDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
#                  asBackbone=False):
#         super(SeqDyReFConv, self).__init__()
#         self.channels = channels
#         self.asBackbone = asBackbone
#
#         with self.name_scope():
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             # self.attention.add(nn.BatchNorm())
#             # if useReLU:
#             #     self.attention.add(nn.Activation('relu'))
#             self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                          dilation=act_dilation, padding=act_dilation,
#                                          groups=channels))
#             self.attention.add(nn.BatchNorm())
#             # if useReLU:
#             #     self.attention.add(nn.Activation('relu'))
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             if not asBackbone:
#                 # self.attention.add(nn.BatchNorm())
#                 self.attention.add(nn.Activation('sigmoid'))
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = F.concat(x3, x5, dim=1)
#         xa = self.attention(xa)
#
#         if self.asBackbone:
#             xs = xa
#         else:
#             xs = x3 * xa + x5 * (1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
# class SK_SeqDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
#                  asBackbone=False):
#         super(SK_SeqDyReFConv, self).__init__()
#         self.channels = channels
#         self.asBackbone = asBackbone
#
#         with self.name_scope():
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='attention')
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             # self.attention.add(nn.BatchNorm())
#             # if useReLU:
#             #     self.attention.add(nn.Activation('relu'))
#             self.attention.add(nn.Conv2D(channels, kernel_size=3, strides=1,
#                                          dilation=act_dilation, padding=act_dilation, groups=channels))
#             self.attention.add(nn.BatchNorm())
#             # self.attention.add(nn.Activation('relu'))
#             # if useReLU:
#             #     self.attention.add(nn.Activation('relu'))
#             # self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             if not asBackbone:
#                 # self.attention.add(nn.BatchNorm())
#                 self.attention.add(nn.Activation('sigmoid'))
#
#             # self.bn = nn.BatchNorm()
#             # self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         # xa = F.concat(x3, x5, dim=1)
#         xa = x3 + x5
#         xa = self.attention(xa)
#
#         if self.asBackbone:
#             xs = xa
#         else:
#             xs = x3 * xa + x5 * (1 - xa)
#
#         # xs = self.bn(xs)
#         # xs = self.relu(xs)
#
#         return xs
#
#
# class SK_LGChaDyReFConv(HybridBlock):
#     def __init__(self, channels=64, dilations=(1, 2), stride=1, act_dilation=1, useReLU=True,
#                  asBackbone=False):
#         super(SK_LGChaDyReFConv, self).__init__()
#         self.channels = channels
#         self.asBackbone = asBackbone
#
#         with self.name_scope():
#             dilation_1 = dilations[0]
#             dilation_2 = dilations[1]
#             self.feature_spatial_3 = nn.HybridSequential(prefix='feature_spatial_3')
#             self.feature_spatial_3.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_1, dilation=dilation_1))
#             self.feature_spatial_3.add(nn.BatchNorm())
#
#             self.feature_spatial_5 = nn.HybridSequential(prefix='feature_spatial_5')
#             self.feature_spatial_5.add(nn.Conv2D(channels, kernel_size=3, strides=stride,
#                                                  padding=dilation_2, dilation=dilation_2))
#             self.feature_spatial_5.add(nn.BatchNorm())
#
#             self.local_att = nn.HybridSequential(prefix='local_att')
#             self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             self.local_att.add(nn.BatchNorm())
#
#             self.global_att = nn.HybridSequential(prefix='global_att')
#             self.global_att.add(nn.GlobalAvgPool2D())
#             self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
#             self.global_att.add(nn.BatchNorm())
#
#             self.sigmoid = nn.Activation('sigmoid')
#
#     def hybrid_forward(self, F, x):
#
#         x3 = self.feature_spatial_3(x)
#         x5 = self.feature_spatial_5(x)
#
#         xa = x3 + x5                  # input of attention
#         xl = self.local_att(xa)       # local attention
#         xg = self.global_att(xa)      # global attention
#         xo = F.broadcast_add(xl, xg)  # output of attention
#         xa = self.sigmoid(xo)
#
#         xs = x3 * xa + x5 * (1 - xa)
#
#         return xs
#
#
