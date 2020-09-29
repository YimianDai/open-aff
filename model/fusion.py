from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class ResGlobChaFuse(HybridBlock):
    def __init__(self, channels=64, asBackbone=False):
        super(ResGlobChaFuse, self).__init__()
        self.asBackbone = asBackbone

        with self.name_scope():

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        wei = self.attention(xa)

        if self.asBackbone:
            xo = wei
        else:
            xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ResLocaChaFuse(HybridBlock):
    def __init__(self, channels=64, asBackbone=False):
        super(ResLocaChaFuse, self).__init__()
        self.asBackbone = asBackbone

        with self.name_scope():

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        wei = self.attention(xa)

        if self.asBackbone:
            xo = wei
        else:
            xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ResGlobLocaChaFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ResGlobLocaChaFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ResLocaLocaChaFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(ResLocaLocaChaFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add((nn.Activation('relu')))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add((nn.Activation('relu')))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ResGlobGlobChaFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(ResGlobGlobChaFuse, self).__init__()

        inter_channels = int(channels // r)

        with self.name_scope():

            # self.local_att = nn.HybridSequential(prefix='local_att')
            # self.local_att.add(nn.GlobalAvgPool2D())
            # self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.local_att.add(nn.BatchNorm())
            #
            # self.global_att = nn.HybridSequential(prefix='global_att')
            # self.global_att.add(nn.GlobalAvgPool2D())
            # self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att.add(nn.BatchNorm())

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.GlobalAvgPool2D())
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add((nn.Activation('relu')))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add((nn.Activation('relu')))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class DirectAddFuse(HybridBlock):
    def __init__(self):
        super(DirectAddFuse, self).__init__()

    def hybrid_forward(self, F, x, residual):

        xo = x + residual

        return xo


class ConcatFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ConcatFuse, self).__init__()

        with self.name_scope():

            self.fuse = nn.HybridSequential(prefix='fuse')
            self.fuse.add(nn.BatchNorm())
            self.fuse.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))

    def hybrid_forward(self, F, x, residual):

        xi = F.concat(x, residual, dim=1)
        xo = self.fuse(xi)

        return xo


class ResGlobLocaforGlobLocaChaFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(ResGlobLocaforGlobLocaChaFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

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

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig1(xlg)

        xi = F.broadcast_mul(x, wei) + F.broadcast_mul(residual, 1-wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = F.broadcast_add(xl2, xg2)
        wei2 = self.sig2(xlg2)
        xo = F.broadcast_mul(x, wei2) + F.broadcast_mul(residual, 1-wei2)

        return xo


class ResGlobforGlobChaFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ResGlobforGlobChaFuse, self).__init__()
        # inter_channels = int(channels // 4)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            # self.global_att.add(nn.Activation('relu'))
            # self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att.add(nn.BatchNorm())

            self.global_att2 = nn.HybridSequential(prefix='global_att2')
            self.global_att2.add(nn.GlobalAvgPool2D())
            self.global_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att2.add(nn.BatchNorm())
            # self.global_att2.add(nn.Activation('relu'))
            # self.global_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att2.add(nn.BatchNorm())

            self.sig1 = nn.Activation('sigmoid')
            self.sig2 = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xg = self.global_att(xa)
        wei = self.sig1(xg)

        xi = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)
        xg2 = self.global_att2(xi)
        wei2 = self.sig2(xg2)
        xo = 2 * F.broadcast_mul(x, wei2) + 2 * F.broadcast_mul(residual, 1-wei2)

        return xo


class ResLocaforLocaChaFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ResLocaforLocaChaFuse, self).__init__()
        # inter_channels = int(channels // 4)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            # self.global_att.add(nn.Activation('relu'))
            # self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att.add(nn.BatchNorm())

            self.local_att2 = nn.HybridSequential(prefix='local_att2')
            self.local_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att2.add(nn.BatchNorm())
            # self.global_att2.add(nn.Activation('relu'))
            # self.global_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att2.add(nn.BatchNorm())

            self.sig1 = nn.Activation('sigmoid')
            self.sig2 = nn.Activation('sigmoid')

            # self.sig1 = nn.Activation('relu')
            # self.sig2 = nn.Activation('relu')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xg = self.local_att(xa)
        wei = self.sig1(xg)

        xi = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)
        xg2 = self.local_att2(xi)
        wei2 = self.sig2(xg2)
        xo = 2 * F.broadcast_mul(x, wei2) + 2 * F.broadcast_mul(residual, 1-wei2)

        return xo


# A(Y) * X + Y
class AYforXplusYAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(AYforXplusYAddFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = F.broadcast_mul(wei, residual) + x

        return xo


# X + A(Y) * Y
class XplusAYforYAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(XplusAYforYAddFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = residual + F.broadcast_mul(wei, x)

        return xo


# A(X+Y)*X + Y
class AXYforXplusYAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(AXYforXplusYAddFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xi = x + residual
        xl = self.local_att(xi)
        xg = self.global_att(xi)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = F.broadcast_mul(wei, residual) + x

        return xo


# A(X+Y) * (X+Y)
class AXYforXYAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(AXYforXYAddFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xi = x + residual
        xl = self.local_att(xi)
        xg = self.global_att(xi)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = F.broadcast_mul(wei, xi)

        return xo


# X + A(Y) * Y
class SEFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(SEFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        wei = self.global_att(x)

        xo = residual + F.broadcast_mul(wei, x)

        return xo


# A(Y) * X + Y
class GAUFuse(HybridBlock):
    def __init__(self, channels=64):
        super(GAUFuse, self).__init__()

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        wei = self.global_att(x)

        xo = F.broadcast_mul(wei, residual) + x

        return xo


class ASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(ASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

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


            self.sig = nn.HybridSequential(prefix='sig')
            # self.sig.add(nn.BatchNorm())
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class HalfASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(HalfASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.post.add(nn.BatchNorm())
            self.post.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xg = F.broadcast_add(xl, xg) - xl
        xlg = F.concat(xl, xg, dim=1)
        wei = self.post(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ConcatASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(ConcatASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.post.add(nn.BatchNorm())
            self.post.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xg = F.broadcast_add(xl, xg) - xl
        xlg = F.concat(xl, xg, dim=1)
        wei = self.post(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class DepthASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(DepthASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                         groups=channels))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.HybridSequential(prefix='sig')
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)

        wei = self.sig(xlg)
        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class Depth4ASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(Depth4ASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att1 = nn.HybridSequential(prefix='local_att1')
            self.local_att1.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att1.add(nn.BatchNorm())
            self.local_att1.add(nn.Activation('relu'))
            self.local_att1.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att1.add(nn.BatchNorm())

            self.local_att2 = nn.HybridSequential(prefix='local_att2')
            self.local_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att2.add(nn.BatchNorm())
            self.local_att2.add(nn.Activation('relu'))
            self.local_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att2.add(nn.BatchNorm())


            self.local_att3 = nn.HybridSequential(prefix='local_att3')
            self.local_att3.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att3.add(nn.BatchNorm())
            self.local_att3.add(nn.Activation('relu'))
            self.local_att3.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att3.add(nn.BatchNorm())

            self.local_att4 = nn.HybridSequential(prefix='local_att4')
            self.local_att4.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att4.add(nn.BatchNorm())
            self.local_att4.add(nn.Activation('relu'))
            self.local_att4.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                          groups=channels))
            self.local_att4.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.HybridSequential(prefix='sig')
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl1 = self.local_att1(xa)
        xl2 = self.local_att2(xa)
        xl3 = self.local_att3(xa)
        xl4 = self.local_att4(xa)
        xl12 = F.broadcast_add(xl1, xl2)
        xl123 = F.broadcast_add(xl12, xl3)
        xl1234 = F.broadcast_add(xl123, xl4)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl1234, xg)

        wei = self.sig(xlg)
        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class AddReLUASKCFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(AddReLUASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))

            self.sig = nn.HybridSequential(prefix='sig')
            # self.sig.add(nn.BatchNorm())
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class LCNASKCFuse(HybridBlock):
    def __init__(self, channels=64):
        super(LCNASKCFuse, self).__init__()
        inter_channels = int(channels // 4)

        with self.name_scope():

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


            self.sig = nn.HybridSequential(prefix='sig')
            # self.sig.add(nn.BatchNorm())
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class ConcatASKC(HybridBlock):
    def __init__(self, channels=64):
        super(ConcatASKC, self).__init__()
        inter_channels = int(channels // 4)

        with self.name_scope():

            self.shared_att = nn.HybridSequential(prefix='shared_att')
            self.shared_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.shared_att.add(nn.BatchNorm())
            self.shared_att.add(nn.Activation('relu'))

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            # self.local_att.add(nn.Activation('relu'))
            # self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            # self.global_att.add(nn.Activation('relu'))
            # self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att.add(nn.BatchNorm())


            self.sig = nn.HybridSequential(prefix='sig')
            self.sig.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.sig.add(nn.BatchNorm())
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = self.shared_att(x + residual)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class TwoConcatASKC(HybridBlock):
    def __init__(self, channels=64):
        super(TwoConcatASKC, self).__init__()
        inter_channels = int(channels // 4)

        with self.name_scope():

            self.pre_att = nn.HybridSequential(prefix='shared_att')
            self.pre_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.pre_att.add(nn.BatchNorm())
            self.pre_att.add(nn.Activation('relu'))

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())
            self.local_att.add(nn.Activation('relu'))
            # self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            # self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            # self.global_att.add(nn.BatchNorm())

            self.post_att = nn.HybridSequential(prefix='shared_att')
            self.post_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.post_att.add(nn.BatchNorm())
            self.post_att.add(nn.Activation('sigmoid'))

            # self.sig = nn.HybridSequential(prefix='sig')
            # # self.sig.add(nn.BatchNorm())
            # self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = F.concat(x, residual, dim=1)
        xa = self.pre_att(xa)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.concat(xl, xg, dim=1)
        wei = self.post_att(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo


class SKFuse(HybridBlock):
    def __init__(self, channels=64):
        super(SKFuse, self).__init__()
        inter_channels = int(channels // 4)
        softmax_channels = int(channels * 2)

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())
            self.global_att.add(nn.Activation('relu'))
            self.global_att.add(nn.Conv2D(softmax_channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xg = self.global_att(xa)
        xg = F.reshape(xg, (0, 2, -1, 0))  # (B, 2, C, 1)
        wei = F.softmax(xg, axis=1)

        wei3 = F.slice_axis(wei, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        wei3 = F.reshape(wei3, (0, -1, 1, 1))
        wei5 = F.slice_axis(wei, axis=1, begin=1, end=2)
        wei5 = F.reshape(wei5, (0, -1, 1, 1))

        xo = 2 * F.broadcast_mul(x, wei3) + 2 * F.broadcast_mul(residual, wei5)

        return xo


class RASKCFuse(HybridBlock):
    def __init__(self, channels=64):
        super(RASKCFuse, self).__init__()
        inter_channels = int(channels // 4)

        with self.name_scope():

            # self.pre = nn.HybridSequential(prefix='pre')
            # self.pre.add(nn.BatchNorm())
            # self.pre.add(nn.Activation('relu'))

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

            self.sig = nn.HybridSequential(prefix='sig')
            # self.sig.add(nn.BatchNorm())
            self.sig.add(nn.Activation('sigmoid'))

            # self.pre2 = nn.HybridSequential(prefix='pre2')
            # self.pre2.add(nn.BatchNorm())
            # self.pre2.add(nn.Activation('relu'))

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

            self.sig2 = nn.HybridSequential(prefix='sig2')
            # self.sig2.add(nn.BatchNorm())
            self.sig2.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        # xa = self.pre(x + residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        # xlg = self.pre2(F.broadcast_add(xl, xg))
        xlg = F.broadcast_add(xl, xg)

        xl2 = self.local_att2(xlg)
        xg2 = self.global_att2(xlg)
        xlg2 = F.broadcast_add(xl2, xg2)
        wei2 = self.sig2(xlg2)
        # wei2 = self.sig2(xg2)
        xlg3 = F.broadcast_add(F.broadcast_mul(xl, wei2), F.broadcast_mul(xg, 1-wei2))
        # wei = self.sig(xlg3)
        wei = self.sig(xlg3)
        xo = F.broadcast_mul(x, wei) + F.broadcast_mul(residual, 1-wei)

        return xo
