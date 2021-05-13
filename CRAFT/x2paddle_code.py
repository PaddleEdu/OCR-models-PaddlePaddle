import paddle


class Upconv4_conv(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv4_conv, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), in_channels=192)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding=1, in_channels=64)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        return x6

class Basenet_slice1(paddle.nn.Layer):
    def __init__(self, ):
        super(Basenet_slice1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=64)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv2 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=1, in_channels=64)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=1, in_channels=128)
        self.bn3 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x8 = self.maxpool0(x6)
        x9 = self.conv2(x8)
        x10 = self.bn2(x9)
        x11 = self.relu2(x10)
        x12 = self.conv3(x11)
        x13 = self.bn3(x12)
        return x13

class Upconv2_conv(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv2_conv, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), in_channels=768)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=1, in_channels=256)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        return x6

class Upconv1_conv(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv1_conv, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), in_channels=1536)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=512)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        return x6

class Upconv3_conv(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv3_conv, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=384)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=128)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        return x6

class Basenet_slice2(paddle.nn.Layer):
    def __init__(self, ):
        super(Basenet_slice2, self).__init__()
        self.relu0 = paddle.nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=128)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=256)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
    def forward(self, x0):
        x1 = self.relu0(x0)
        x3 = self.maxpool0(x1)
        x4 = self.conv0(x3)
        x5 = self.bn0(x4)
        x6 = self.relu1(x5)
        x7 = self.conv1(x6)
        x8 = self.bn1(x7)
        x9 = (x8, x1, x1, x1, x1, x1)
        x10, x11, x12, x13, x14, x15 = x9
        return x10, x11, x12, x13, x14, x15

class Basenet_slice4(paddle.nn.Layer):
    def __init__(self, ):
        super(Basenet_slice4, self).__init__()
        self.relu0 = paddle.nn.ReLU()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv1 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
    def forward(self, x0):
        x1 = self.relu0(x0)
        x2 = self.conv0(x1)
        x3 = self.bn0(x2)
        x4 = self.relu1(x3)
        x6 = self.maxpool0(x4)
        x7 = self.conv1(x6)
        x8 = self.bn1(x7)
        x9 = self.relu2(x8)
        x10 = self.conv2(x9)
        x11 = self.bn2(x10)
        x12 = (x11, x1, x1, x1, x1, x1)
        x13, x14, x15, x16, x17, x18 = x12
        return x13, x14, x15, x16, x17, x18

class Basenet_slice3(paddle.nn.Layer):
    def __init__(self, ):
        super(Basenet_slice3, self).__init__()
        self.relu0 = paddle.nn.ReLU()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=256)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv1 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=256)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
    def forward(self, x0):
        x1 = self.relu0(x0)
        x2 = self.conv0(x1)
        x3 = self.bn0(x2)
        x4 = self.relu1(x3)
        x6 = self.maxpool0(x4)
        x7 = self.conv1(x6)
        x8 = self.bn1(x7)
        x9 = self.relu2(x8)
        x10 = self.conv2(x9)
        x11 = self.bn2(x10)
        x12 = (x11, x1, x1, x1, x1, x1)
        x13, x14, x15, x16, x17, x18 = x12
        return x13, x14, x15, x16, x17, x18

class Basenet_slice5(paddle.nn.Layer):
    def __init__(self, ):
        super(Basenet_slice5, self).__init__()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=1, padding=1)
        self.conv0 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(3, 3), padding=6, dilation=6, in_channels=512)
        self.conv1 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), in_channels=1024)
    def forward(self, x1, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19):
        x2 = self.maxpool0(x1)
        x3 = self.conv0(x2)
        x4 = self.conv1(x3)
        x20 = (x4, x1, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19)
        x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37 = x20
        return x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37

class vgg16_bn(paddle.nn.Layer):
    def __init__(self, ):
        super(vgg16_bn, self).__init__()
        self.basenet_slice10 = Basenet_slice1()
        self.basenet_slice20 = Basenet_slice2()
        self.basenet_slice30 = Basenet_slice3()
        self.basenet_slice40 = Basenet_slice4()
        self.basenet_slice50 = Basenet_slice5()
    def forward(self, x0):
        x1 = self.basenet_slice10(x0)
        x2,x3,x4,x5,x6,x7 = self.basenet_slice20(x1)
        x8,x9,x10,x11,x12,x13 = self.basenet_slice30(x2)
        x14,x15,x16,x17,x18,x19 = self.basenet_slice40(x8)
        x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36 = self.basenet_slice50(x14, x15, x16, x17, x18, x19, x9, x10, x11, x12, x13, x3, x4, x5, x6, x7)
        return x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36

class double_conv0(paddle.nn.Layer):
    def __init__(self, ):
        super(double_conv0, self).__init__()
        self.upconv4_conv0 = Upconv4_conv()
    def forward(self, x0):
        x1 = self.upconv4_conv0(x0)
        return x1

class double_conv1(paddle.nn.Layer):
    def __init__(self, ):
        super(double_conv1, self).__init__()
        self.upconv3_conv0 = Upconv3_conv()
    def forward(self, x0):
        x1 = self.upconv3_conv0(x0)
        return x1

class double_conv2(paddle.nn.Layer):
    def __init__(self, ):
        super(double_conv2, self).__init__()
        self.upconv1_conv0 = Upconv1_conv()
    def forward(self, x0):
        x1 = self.upconv1_conv0(x0)
        return x1

class double_conv3(paddle.nn.Layer):
    def __init__(self, ):
        super(double_conv3, self).__init__()
        self.upconv2_conv0 = Upconv2_conv()
    def forward(self, x0):
        x1 = self.upconv2_conv0(x0)
        return x1

class Conv_cls(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv_cls, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding=1, in_channels=32)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding=1, in_channels=32)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=16, kernel_size=(3, 3), padding=1, in_channels=32)
        self.relu2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(out_channels=16, kernel_size=(1, 1), in_channels=16)
        self.relu3 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(out_channels=2, kernel_size=(1, 1), in_channels=16)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        x5 = self.conv2(x4)
        x6 = self.relu2(x5)
        x7 = self.conv3(x6)
        x8 = self.relu3(x7)
        x9 = self.conv4(x8)
        return x9

class CRAFT(paddle.nn.Layer):
    def __init__(self, ):
        super(CRAFT, self).__init__()
        self.vgg16_bn0 = vgg16_bn()
        self.double_conv20 = double_conv2()
        self.double_conv30 = double_conv3()
        self.double_conv10 = double_conv1()
        self.double_conv00 = double_conv0()
        self.conv_cls0 = Conv_cls()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17 = self.vgg16_bn0(x0)
        x18 = [x1, x2]
        x19 = paddle.concat(x=x18, axis=1)
        x20 = self.double_conv20(x19)
        x21 = paddle.nn.functional.interpolate(x=x20, size=[92, 160], mode='bilinear')
        x22 = [x21, x7]
        x23 = paddle.concat(x=x22, axis=1)
        x24 = self.double_conv30(x23)
        x25 = paddle.nn.functional.interpolate(x=x24, size=[184, 320], mode='bilinear')
        x26 = [x25, x12]
        x27 = paddle.concat(x=x26, axis=1)
        x28 = self.double_conv10(x27)
        x29 = paddle.nn.functional.interpolate(x=x28, size=[368, 640], mode='bilinear')
        x30 = [x29, x17]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.double_conv00(x31)
        x33 = self.conv_cls0(x32)
        x34 = paddle.transpose(x=x33, perm=[0, 2, 3, 1])
        x35 = (x34, x32)
        return x35
