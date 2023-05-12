# File: resnet.py
#
# Author: Thomas Bandy
#
# This file contains the implementation of the ResNet18 model and the Residual Block module in PyTorch.
# The ResNet18 model is a 4-layer variant with 18 convolutional layers.
#
# References:
# - https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
# - https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
#
# The ResNet18 model is based on the ResNet architecture proposed in the above paper,
# with modifications for the ResNet18 variant.
#
# The Residual Block module is a building block used in the ResNet18 model,
# consisting of convolutional layers, batch normalization, and skip connections.
#
# Date: May 12, 2023

import torch
from torch import nn


class ResNet18(nn.Module):

    def __init__(self, in_channels, resblock, outputs=1000):
        '''A 4-layer ResNet model with 18 convolutional layers.

            Args:
                in_channels (int): Number of input channels.
                resblock (class): Residual block class to be used in the network.
                outputs (int): Number of output classes. Defaults to 1000.

            Attributes:
                layer0 (nn.Sequential): First layer of the network, including initial convolution, pooling, batch normalization, and ReLU activation.
                layer1 (nn.Sequential): Second layer of the network, consisting of two residual blocks.
                layer2 (nn.Sequential): Third layer of the network, consisting of two residual blocks with downsample in the first block.
                layer3 (nn.Sequential): Fourth layer of the network, consisting of two residual blocks with downsample in the first block.
                layer4 (nn.Sequential): Fifth layer of the network, consisting of two residual blocks with downsample in the first block.
                gap (nn.AdaptiveAvgPool2d): Global average pooling layer.
                fc (nn.Linear): Fully connected layer for classification.

            Methods:
                forward(input): Performs the forward pass of the ResNet18 model.
        '''

        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(32, 64, downsample=True),
            resblock(64, 64, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(256, outputs)

    def forward(self, input):
        ''' Performs the forward pass of the ResNet18 model.

            Args:
                input (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after applying the forward pass.
        '''
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        input = self.fc(input)

        return input


class resblock(nn.Module):
    '''A residual block module for a convolutional neural network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample (bool): Indicates if the input spatial dimensions need to be downsampled.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        shortcut (nn.Sequential): Shortcut connection used for downsampling.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.

    Methods:
        forward(input): Performs the forward pass of the residual block.
    '''

    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        '''Performs the forward pass of the residual block.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        '''

        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)