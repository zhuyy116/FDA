import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *


def get_cls(name, planes):
    if name is None:
        return None
    elif name.lower() == 'fda':
        return FDA(planes)
    elif name.lower() == 'fda_plus':
        return FDA_plus(planes)
    else:
        raise ValueError(name + " class do not exist!!!")


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, att_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.att = get_cls(att_layer, planes)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.att is not None:
            out = self.att(out)

        if x.size(1) != out.size(1):
            y = F.pad(x[:, :, ::2, ::2], (
                0, 0, 0, 0, (out.size(1) - x.size(1)) // 2, (out.size(1) - x.size(1)) - (out.size(1) - x.size(1)) // 2),
                      "constant", 0)
            out += y
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, att_type=None):
        super(resnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, att_type=att_type)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, att_type=att_type)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, att_type=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, att_type))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_cifar(network_type, depth, num_classes, att_later=None):
    assert network_type in ["cifar10", "cifar100"], "network type should be cifar10 or cifar100"
    assert int(depth[6:]) in [20, 32, 44, 56, 110, 1202], 'network depth should be 20, 32, 44, 56, 110, 1202'

    if int(depth[6:]) == 20:
        model = resnet(BasicBlock, [3, 3, 3], num_classes, att_later)
    elif int(depth[6:]) == 32:
        model = resnet(BasicBlock, [5, 5, 5], num_classes, att_later)
    elif int(depth[6:]) == 44:
        model = resnet(BasicBlock, [7, 7, 7], num_classes, att_later)
    elif int(depth[6:]) == 56:
        model = resnet(BasicBlock, [9, 9, 9], num_classes, att_later)
    elif int(depth[6:]) == 110:
        model = resnet(BasicBlock, [18, 18, 18], num_classes, att_later)
    elif int(depth[6:]) == 1202:
        model = resnet(BasicBlock, [200, 200, 200], num_classes, att_later)
    else:
        model = None

    return model
