from torch import nn
import torch
from torchsummary import summary


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):#groups=inchannel:pw
        padding = (kernel_size - 1) // 2#k=3,p=1;k=1,p=0
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, t):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * t
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if t != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels=1,num_classes=1000):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        D0 = []
        D1 = []
        D2=[]
        D3=[]
        D4=[]
        D5=[]
                                                #output
        D0.append(ConvBNReLU(in_channels, 3, stride=1))#32, 224, 224

        D1.append(ConvBNReLU(3, 32, stride=1))#32, 224, 224
        D1.append(block(32, 16, stride=1, t=1))#16, 224, 224

        D2.append(block(16, 24, stride=2, t=6))#24, 112, 112
        D2.append(block(24, 24, stride=1, t=6))#24, 112, 112

        D3.append(block(24, 32, stride=2, t=6))#32, 56, 56
        D3.append(block(32, 32, stride=1, t=6))#32, 56, 56
        D3.append(block(32, 32, stride=1, t=6))#32, 56, 56

        D4.append(block(32, 64, stride=2, t=6))#64, 28, 28
        D4.append(block(64, 64, stride=1, t=6))#64, 28, 28
        D4.append(block(64, 64, stride=1, t=6))#64, 28, 28
        D4.append(block(64, 64, stride=1, t=6))#64, 28, 28

        D5.append(block(64, 96, stride=2, t=6))#96, 28, 28
        D5.append(block(96, 96, stride=1, t=6))#96, 28, 28
        # D4.append(block(96, 96, stride=1, t=6))#96, 28, 28
        #
        # D5.append(block(96, 160, stride=2, t=6))#160, 14, 14
        # D5.append(block(160, 160, stride=1, t=6))#160, 14, 14
        # D5.append(block(160, 160, stride=1, t=6))#160, 14, 14
        #
        # D5.append(block(160, 320, stride=1, t=6))#320, 14, 14

        self.D0 = nn.Sequential(*D0)
        self.D1 = nn.Sequential(*D1)
        self.D2 = nn.Sequential(*D2)
        self.D3 = nn.Sequential(*D3)
        self.D4 = nn.Sequential(*D4)
        self.D5 = nn.Sequential(*D5)
        # self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        # self.C1 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(256, num_classes)
        # )
        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')#kaiming 正态分布
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)#正态分布初始化
        #         nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = self.D0(x)
        x1 = self.D1(x0)#3,512,512 -> 16, 224, 224
        x2 = self.D2(x1)#16, 224, 224 -> 24, 112, 112
        x3 = self.D3(x2)#24, 112, 112 -> 32, 56, 56
        x4 = self.D4(x3)#32, 56, 56 -> 96, 28, 28
        x5 = self.D5(x4)#96, 28, 28 -> 320, 14, 14
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # # x5 = self.conv(x5)
        # x = self.avgpool(x5)#256,16,16 -> 256,1,1
        # x = torch.flatten(x, 1)#256,1,1 -> 256
        # x = self.C1(x)#256 -> 5
        # x = self.softmax(x)
        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    model = MobileNetV2(num_classes=2)
    # m = model.state_dict()
    # state_dict = torch.load("../mobilenet_v2-b0353104.pth",map_location='cpu')
    # for key,v in state_dict.items():
    # model.load_state_dict(state_dict,)
    # print(model)
    # model = MobileNetV2(num_classes=2)
    summary(model, (1, 224, 224), device="cpu")
    # print(model)
