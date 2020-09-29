from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

from .activation import MSSeqATAC, SeqATAC, ChaATAC, SpaATAC, xUnit


class LearnedCell(HybridBlock):
    def __init__(self, dial1, dial2, channels=64):
        super(LearnedCell, self).__init__()
        self.channels = channels
        with self.name_scope():

            self.feature_spatial_1 = nn.HybridSequential(prefix='feature_spatial_1')
            self.feature_spatial_1.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dial1, dilation=dial1))
            # self.feature_spatial_1.add(nn.BatchNorm())

            self.feature_spatial_2 = nn.HybridSequential(prefix='feature_spatial_2')
            self.feature_spatial_2.add(nn.Conv2D(channels, kernel_size=3, strides=1,
                                                 padding=dial2, dilation=dial2))
            # self.feature_spatial_2.add(nn.BatchNorm())

            self.feature_channel = nn.HybridSequential(prefix='feature_channel')
            self.feature_channel.add(nn.BatchNorm())
            self.feature_channel.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                                               padding=0))
            self.feature_channel.add(nn.BatchNorm())
            self.feature_channel.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):

        x1 = self.feature_spatial_1(x)
        x2 = self.feature_spatial_2(x)
        xc = F.concat(x1, x2, dim=1)
        xc = self.feature_channel(xc)

        return xc



