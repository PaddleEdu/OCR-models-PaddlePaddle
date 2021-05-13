"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import paddle
import paddle.nn as nn
from basenet.vgg16_bn import init_weights


class RefineNet(nn.Layer):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = nn.Sequential(
            nn.Conv2D(34, 64, kernel_size=3, padding=1), nn.BatchNorm2D(64), nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=3, padding=1), nn.BatchNorm2D(64), nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=3, padding=1), nn.BatchNorm2D(64), nn.ReLU()
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 128, kernel_size=1), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 128, kernel_size=1), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 128, kernel_size=1), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 128, kernel_size=1), nn.BatchNorm2D(128), nn.ReLU(),
            nn.Conv2D(128, 1, kernel_size=1)
        )

        init_weights(self.last_conv.sublayers())
        init_weights(self.aspp1.sublayers())
        init_weights(self.aspp2.sublayers())
        init_weights(self.aspp3.sublayers())
        init_weights(self.aspp4.sublayers())

    def forward(self, y, upconv4):
        refine = paddle.concat([y.transpose([0,3,1,2]), upconv4], axis=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        #out = paddle.add([aspp1, aspp2, aspp3, aspp4], axis=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.transpose([0, 2, 3, 1])  # , refine.transpose([0,2,3,1])
