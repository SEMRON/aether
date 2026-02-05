import torch as t
import torch.nn as nn
import torch.nn.functional as F

from hivemind.moe.server.layers.custom_experts import register_expert_class

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def head_sample_input(batch_size, num_channels: int, img_size: int):
    return (t.empty((batch_size, num_channels, img_size, img_size)),)

def resnet18_tail_sample_input(batch_size: int, img_size: int):
    """Sample input for `resnet18.tail` given the original image size.

    The `resnet18.head` in this repo keeps spatial size in conv1/layer1, then downsamples once in layer2 (stride=2).
    """
    s = (img_size + 1) // 2
    return t.empty((batch_size, 128, s, s))


def resnet50_tail_sample_input(batch_size: int, img_size: int):
    """Sample input for `resnet50.tail` given the original image size.

    With the ImageNet-style stem (conv1 stride=2 + maxpool stride=2) and layer2 stride=2 in `resnet50.head`,
    the spatial size is downsampled by ~8x: 224 -> 112 -> 56 -> 28.
    """
    s1 = (img_size + 1) // 2  # conv1, stride=2
    s2 = (s1 + 1) // 2       # maxpool, stride=2
    s3 = (s2 + 1) // 2       # layer2, stride=2 (bottleneck conv2)
    return t.empty((batch_size, 512, s3, s3))


@register_expert_class("resnet18.full", head_sample_input)
class ResNet18Full(nn.Module):
    def __init__(self, *args, num_classes: int, num_channels: int, **kwargs):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels)
        
    def forward(self, x):
        return self.resnet(x)

@register_expert_class("resnet50.full", head_sample_input)
class ResNet50Full(nn.Module):
    def __init__(self, *args, num_classes: int = 10, num_channels: int = 3, model_config=None, **kwargs):
        if model_config is not None:
            num_classes = getattr(model_config, "num_classes", num_classes)
            extra = getattr(model_config, "extra", {}) or {}
            num_channels = extra.get("num_channels", num_channels)
        super().__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels)
        
    def forward(self, x):
        return self.resnet(x)


@register_expert_class("resnet101.full", head_sample_input)
class ResNet101Full(nn.Module):
    def __init__(self, *args, num_classes: int = 10, num_channels: int = 3, model_config=None, **kwargs):
        if model_config is not None:
            num_classes = getattr(model_config, "num_classes", num_classes)
            extra = getattr(model_config, "extra", {}) or {}
            num_channels = extra.get("num_channels", num_channels)
        super().__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_channels=num_channels)
        
    def forward(self, x):
        return self.resnet(x)


@register_expert_class("resnet18.head", head_sample_input)
class Resnet18Front(nn.Module):
    def __init__(self, *args, num_channels: int, **kwargs):
        super().__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

@register_expert_class("resnet18.tail", resnet18_tail_sample_input)
class Resnet18Back(nn.Module):
    def __init__(self, *args, num_classes: int, **kwargs):
        super().__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]

        self.in_planes = 128
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@register_expert_class("resnet50.head", head_sample_input)
class Resnet50Head(nn.Module):
    def __init__(self, *args, num_channels: int, **kwargs):
        super().__init__()
        block = Bottleneck
        num_blocks = [3, 4, 6, 3]

        self.in_planes = 64
        # ImageNet-style stem for ResNet-50 split (keeps activation sizes reasonable for 224x224 inputs)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

@register_expert_class("resnet50.tail", resnet50_tail_sample_input)
class Resnet50Tail(nn.Module):
    def __init__(self, *args, num_classes: int, **kwargs):
        super().__init__()
        block = Bottleneck
        num_blocks = [3, 4, 6, 3]

        # `resnet50.head` ends after layer2, whose output channels are 128 * expansion = 512
        self.in_planes = 512
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out