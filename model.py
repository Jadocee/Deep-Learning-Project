# File: resnet.py
#
# Author: Thomas Bandy
#
import torch
import torch.nn as nn
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
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

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
        input = input.view(input.size(0), -1)
        input = self.fc(input)

        return input


class Resblock(nn.Module):
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




VGG11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ]
class VGGnet(nn.Module):
    def __init__(self, in_channels, outputs=1000):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.make_conv_layers(VGG11)

        self.fcs = nn.Sequential(
        nn.Linear(41472, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, outputs)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        x = x.view(41472,1)
        return x

    def make_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]

                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)




