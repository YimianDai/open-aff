from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class DirectAddFuse(HybridBlock):
    def __init__(self):
        super(DirectAddFuse, self).__init__()

    def hybrid_forward(self, F, x, residual):

        xo = x + residual

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
            self.sig.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo
