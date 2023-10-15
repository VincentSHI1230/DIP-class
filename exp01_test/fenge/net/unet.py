import torch
import torch.nn as nn
from torchsummary import summary
from net.mobilenet import MobileNetV2
from typing import Dict
import torch.nn.functional as F
import os

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1):#groups=inchannel:pw
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):#groups=inchannel:pw
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cbn = nn.Sequential(ConvBNReLU(in_channel, out_channel, kernel_size=3))
        self.cbn1 = nn.Sequential(ConvBNReLU(out_channel*2, out_channel, kernel_size=3),
                                  # nn.Dropout(d),
                                  ConvBNReLU(out_channel, out_channel, kernel_size=3))

    def forward(self, inputs1, inputs2):
        # print(inputs1.shape)
        # print(inputs2.shape)
        b = self.up(inputs1)#图片大小加倍
        # print(b.shape)
        outputs = self.cbn(b)#inputs1降维到与input2一样
        # print(outputs.shape)
        outputs = torch.cat([inputs2, outputs], dim=1)#拼接
        # print(outputs.shape)
        outputs = self.cbn1(outputs)
        # print(outputs.shape)
        return outputs


class UNet(nn.Module):
    def __init__(self,in_channels=1, num_classes=2):
        super(UNet, self).__init__()
        self.mobilenet = MobileNetV2(in_channels,num_classes)
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.conv = ConvBNReLU(256,128)
        # self.convt = nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2, padding=0)
        # self.up1 = Up(320, 96)
        self.up1 = Up(96, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 24)
        self.up4 = Up(24, 16)
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')#kaiming 正态分布
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1, x2, x3, x4, x5 = self.mobilenet.forward(x)
        # x = self.convt(x5)
        # print(x5.shape)
        # print(x4.shape)
        # print(x3.shape)
        # print(x2.shape)
        # print(x1.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits

    def freeze_backbone(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    model = UNet(num_classes=3)
    model_dict = model.state_dict()                                    # 取出自己网络的参数字典
    pretrained_dict = torch.load("../mobilenet_v2-b0353104.pth",map_location='cpu')# 加载预训练网络的参数字典
    # print(pretrained_dict)
    #  取出预训练网络的参数字典
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 1
    # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
    for k, v in model_dict.items():
        # print('{},{}'.format(v.size(),pretrained_dict[keys[i]].size()))
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            #print(model_dict[k])
            i = i + 1
    # print(i)
    model.load_state_dict(model_dict)
    # print(model.state_dict())
    # model_dict = model.state_dict()
    # print(model_dict.items())
    summary(model, (1, 512, 512), device="cpu")
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.3fM" % (total / 1e6))
