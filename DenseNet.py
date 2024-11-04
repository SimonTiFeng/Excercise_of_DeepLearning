import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(in_channels + i * out_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            y = layer(x)
            x = torch.cat([x, y], dim=1)
        return x

def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


imput_layer = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels,growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]

blocks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blocks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        trans_block = transition_block(num_channels, num_channels // 2)
        num_channels //= 2

dense_net = nn.Sequential(
    imput_layer,
    *blocks,
    nn.BatchNorm2d(num_channels),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(num_channels, 10)
)

