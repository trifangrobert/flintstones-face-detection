import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential()
        if input_channels != output_channels:
            # trick from inception architecture
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        

    def forward(self, x):
        output = self.activation(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        return self.activation(output)

class ResNet(nn.Module):
    def __init__(self, block_type, block_configuration, num_classes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential(
            self.make_resnet_layer(block_type, 64, 64, block_configuration[0], stride=1),
            self.make_resnet_layer(block_type, 64, 128, block_configuration[1], stride=2),
            self.make_resnet_layer(block_type, 128, 256, block_configuration[2], stride=2),
            self.make_resnet_layer(block_type, 256, 512, block_configuration[3], stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def make_resnet_layer(self, block_type, input_channels, output_channels, num_blocks, stride):
        layers = []
        layers.append(block_type(input_channels, output_channels, stride))
        for i in range(num_blocks - 1):
            layers.append(block_type(output_channels, output_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.maxpool(output)
        output = self.layers(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
    
def ResNet18(num_classes):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    # dummy data
    x = torch.rand((1, 3, 128, 128))
    model = ResNet18(5)
    y = model(x)
    print(y.shape)