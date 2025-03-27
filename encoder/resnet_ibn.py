import torch
import torch.nn as nn
import torch.nn.functional as F

class IBN(nn.Module):
    """Instance-Batch Normalization Block"""
    def __init__(self, channels):
        super(IBN, self).__init__()
        half1 = int(channels * 0.5)
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(channels - half1)

    def forward(self, x):
        split = torch.split(x, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        out = torch.cat([self.IN(split[0]), self.BN(split[1])], dim=1)
        return out

class ResidualIBN(nn.Module):
    """ResNet Residual Block with IBN"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualIBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = IBN(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    """Standard ResNet-50 Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class GeMPooling(nn.Module):
    """Global Efficient Mean Pooling"""
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)

class ResNetIBN(nn.Module):
    def __init__(self):
        super(ResNetIBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust input size to (84, 216)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResidualIBN, 64, 128, 2, stride=1)
        self.layer2 = self._make_layer(ResidualIBN, 128, 256, 2, stride=1)
        self.layer3 = self._make_layer(ResidualIBN, 256, 512, 2, stride=2)
        self.layer4 = self._make_layer(ResidualIBN, 512, 1024, 2, stride=2)

        self.global_pool = GeMPooling()

        # self.embedding_head = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048)
        # )
        self.embedding_head = nn.Linear(1024, 2048)


    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        # print(f"Conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"Maxpool: {x.shape}")

        x = self.layer1(x)
        # print(f"Layer1: {x.shape}")
        x = self.layer2(x)
        # print(f"Layer2: {x.shape}")
        x = self.layer3(x)
        # print(f"Layer3: {x.shape}")
        x = self.layer4(x)
        # print(f"Layer4: {x.shape}")

        x = self.global_pool(x).view(x.size(0), -1)
        x = self.embedding_head(x)
        # print(f"Global Pool: {x.shape}")
        return x