# File: alexnet.py
#
# Author: Thomas Bandy
#
# This file contains the implementation of the AlexNet model in PyTorch.
# AlexNet is a convolutional neural network model introduced by Alex Krizhevsky et al.
# This implementation follows the original architecture with minor modifications.
#
# References:
# - "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
# - Code implementation by Thomas Bandy: https://github.com/dansuh17/alexnet-pytorch
# - Blog post on AlexNet in PyTorch by Thomas Bandy: https://blog.paperspace.com/alexnet-pytorch/
#
# Date: May 22, 2023

from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet is a convolutional neural network model introduced by Alex Krizhevsky et al.
    This implementation follows the original architecture with minor modifications.

    Args:
        outputs (int): Number of output classes. Default is 10.

    Attributes:
        layer1 (nn.Sequential): First convolutional layer followed by batch normalization,
            ReLU activation, and max pooling.
        layer2 (nn.Sequential): Second convolutional layer followed by batch normalization,
            ReLU activation, and max pooling.
        layer3 (nn.Sequential): Third convolutional layer followed by batch normalization
            and ReLU activation.
        layer4 (nn.Sequential): Fourth convolutional layer followed by batch normalization
            and ReLU activation.
        layer5 (nn.Sequential): Fifth convolutional layer followed by batch normalization,
            ReLU activation, and max pooling.
        fc (nn.Sequential): Fully connected layers with dropout and ReLU activation.

        fc1 (nn.Sequential): Fully connected layers with dropout and ReLU activation.

        fc2 (nn.Sequential): Fully connected layer for final classification.

    Methods:
        forward(x): Performs forward pass of the input through the network.

    """

    def __init__(self, outputs=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(9216, 4096), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, outputs))

    def forward(self, x):
        """
        Performs forward pass of the input through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).

        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
