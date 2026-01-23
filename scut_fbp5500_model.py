"""PyTorch model definitions for SCUT-FBP5500 pretrained weights.

Adapted from https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
(trained_models_for_pytorch/Nets.py).
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn


def _bn_relu(inplanes: int) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))


def _bn_relu_pool(inplanes: int, kernel_size: int = 3, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
    )


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.relu_pool1 = _bn_relu_pool(inplanes=96)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2, groups=2, bias=False)
        self.relu_pool2 = _bn_relu_pool(inplanes=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu3 = _bn_relu(inplanes=384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu4 = _bn_relu(inplanes=384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu_pool5 = _bn_relu_pool(inplanes=256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, groups=2, bias=False)
        self.relu6 = _bn_relu(inplanes=256)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu_pool1(x)
        x = self.conv2(x)
        x = self.relu_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu_pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return x


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None) -> None:
        super().__init__()
        group = OrderedDict()
        group["conv1"] = _conv3x3(inplanes, planes, stride)
        group["bn1"] = nn.BatchNorm2d(planes)
        group["relu1"] = nn.ReLU(inplace=True)
        group["conv2"] = _conv3x3(planes, planes)
        group["bn2"] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(group)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int = 1000) -> None:
        self.inplanes = 64
        super().__init__()

        group = OrderedDict()
        group["conv1"] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        group["bn1"] = nn.BatchNorm2d(64)
        group["relu1"] = nn.ReLU(inplace=True)
        group["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(group)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.group2 = nn.Sequential(
            OrderedDict([("fullyconnected", nn.Linear(512 * block.expansion, num_classes))])
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight.data)
                torch.nn.init.constant_(module.bias.data, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)
        return x
