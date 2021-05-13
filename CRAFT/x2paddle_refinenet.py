import paddle


class Last_conv(paddle.nn.Layer):
    def __init__(self, ):
        super(Last_conv, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=34)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=64)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=64)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.relu2(x8)
        return x9

class Aspp2(paddle.nn.Layer):
    def __init__(self, ):
        super(Aspp2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=12, dilation=12, in_channels=64)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=1, kernel_size=(1, 1), in_channels=128)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        return x7

class Aspp1(paddle.nn.Layer):
    def __init__(self, ):
        super(Aspp1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=6, dilation=6, in_channels=64)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=1, kernel_size=(1, 1), in_channels=128)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        return x7

class Aspp4(paddle.nn.Layer):
    def __init__(self, ):
        super(Aspp4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=24, dilation=24, in_channels=64)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=1, kernel_size=(1, 1), in_channels=128)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        return x7

class Aspp3(paddle.nn.Layer):
    def __init__(self, ):
        super(Aspp3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=18, dilation=18, in_channels=64)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), in_channels=128)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(out_channels=1, kernel_size=(1, 1), in_channels=128)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        return x7

class RefineNet(paddle.nn.Layer):
    def __init__(self, ):
        super(RefineNet, self).__init__()
        self.last_conv0 = Last_conv()
        self.aspp10 = Aspp1()
        self.aspp20 = Aspp2()
        self.aspp30 = Aspp3()
        self.aspp40 = Aspp4()
    def forward(self, x0, x1):
        x0 = paddle.to_tensor(data=x0)
        x1 = paddle.to_tensor(data=x1)
        x2 = paddle.transpose(x=x0, perm=[0, 3, 1, 2])
        x3 = [x2, x1]
        x4 = paddle.concat(x=x3, axis=1)
        x5 = self.last_conv0(x4)
        x6 = self.aspp10(x5)
        x7 = self.aspp20(x5)
        x8 = self.aspp30(x5)
        x9 = self.aspp40(x5)
        x10 = x6 + x7
        x11 = x10 + x8
        x12 = x11 + x9
        x13 = paddle.transpose(x=x12, perm=[0, 2, 3, 1])
        return x13

def main(x0, x1):
    # There are 2 inputs.
    # x0: shape-[1, 368, 640, 2], type-float32.
    # x1: shape-[1, 32, 368, 640], type-float32.
    paddle.disable_static()
    params = paddle.load('/content/CRAFT/CRAFT/paddlemodel/model.pdparams')
    model = RefineNet()
    model.set_dict(params)
    model.eval()
    out = model(x0, x1)
    return out