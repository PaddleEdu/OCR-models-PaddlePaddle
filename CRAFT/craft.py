"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from basenet.vgg16_bn import vgg16_bn, init_weights


class double_conv(nn.Layer):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2D(mid_ch),
            nn.ReLU(),
            nn.Conv2D(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Layer):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2D(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2D(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2D(16, 16, kernel_size=1), nn.ReLU(),
            nn.Conv2D(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.sublayers())
        init_weights(self.upconv2.sublayers())
        init_weights(self.upconv3.sublayers())
        init_weights(self.upconv4.sublayers())
        init_weights(self.conv_cls.sublayers())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)
        """ U network """
        y = paddle.concat([sources[0], sources[1]], axis=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].shape[2:], mode='bilinear', align_corners=False)
        y = paddle.concat([y, sources[2]], axis=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].shape[2:], mode='bilinear', align_corners=False)
        y = paddle.concat([y, sources[3]], axis=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].shape[2:], mode='bilinear', align_corners=False)
        y = paddle.concat([y, sources[4]], axis=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.transpose([0,2,3,1]), feature

if __name__ == '__main__':
    model = CRAFT(pretrained=True)
    output, _ = model(paddle.randn(1, 3, 768, 768))
    print(output.shape)
