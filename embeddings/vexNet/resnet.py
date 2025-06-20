# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

# 11 layer resnet with 3 residual blocks

"""
The residualBlock class defines a single residual block with two convolutional layers
and a skip connection, as described in the resnet paper. The resnet class defines the entire network architecture,
which consists of a single convolutional layer, three sets of residual blocks
(each set containing three blocks), a max pooling layer, and a fully connected output layer.
"""

"""
This CNN has 11 layers.

The layers can be broken down as follows:

Convolutional layer (3x3 kernel, 16 output channels)
Batch normalization layer
ReLU activation layer
3 Residual blocks, each containing:
    Convolutional layer (3x3 kernel, same number of input and output channels)
    Batch normalization layer
    ReLU activation layer
    Convolutional layer (3x3 kernel, same number of input and output channels)
    Batch normalization layer
    Shortcut connection (either an identity mapping or a 1x1 convolutional layer followed by batch normalization layer, depending on the input and output channel dimensions)
    ReLU activation layer
Max pooling layer (2x2 kernel, stride of 2)
Fully connected layer (10 output units, corresponding to the number of classes in the CIFAR-10 dataset)
"""


import torch
import numpy as np
import torch.nn as nn
import h5py


class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.makeLayer(residualBlock, 16, 1, stride=1)
        self.layer2 = self.makeLayer(residualBlock, 32, 1, stride=2)
        self.layer3 = self.makeLayer(residualBlock, 64, 1, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def makeLayer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

        # model = resnet()

        # file = '/Pramana/VexIR2Vec/H5-Training-Data/w7-s1/latest/new_training_data_find_inl_ext_300D_filtered_cfg.h5'
        # hf = h5py.File(file, 'r')
        # cfgs = hf.get('cfgs')[()]

        # ip = cfgs[2]
        # ip=torch.from_numpy(ip).float()
        # ip=ip.unsqueeze(0)
        # ip=ip.unsqueeze(0)

        # print(ip.shape)
        # print(ip.ndim)

        # with torch.no_grad():
        #     model.eval()
        #     features = (model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(ip)))))))
        #     # print(features.shape) #[1,64,n,n]
        #     embedding = features.mean([2,3]).squeeze()

        # print(embedding)
