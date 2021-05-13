from collections import namedtuple
import paddle
import numpy as np
import paddle.nn as nn
from paddle.vision.models import vgg16


def init_weights(sublayers):
    for m in sublayers:
        if isinstance(m, nn.Conv2D):
            m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.XavierUniform())
            if m.bias is not None:
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
        elif isinstance(m, nn.BatchNorm2D):
            m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))
            m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
        elif isinstance(m, nn.Linear):
            m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Normal(0.0, 0.01))
            m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))


class vgg16_bn(paddle.nn.Layer):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=pretrained, batch_norm=True).features
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = paddle.nn.Sequential(
                nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
                nn.Conv2D(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2D(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.sublayers())
            init_weights(self.slice2.sublayers())
            init_weights(self.slice3.sublayers())
            init_weights(self.slice4.sublayers())

        init_weights(self.slice5.sublayers())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.stop_gradient = True

    def forward(self, X):
        paddle.save(self.slice2.state_dict(), 'B.pdparams', 3)
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        print(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
