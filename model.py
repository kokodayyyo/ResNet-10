import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, 1, stride=strides) if use_1conv else None
        self.dropout = nn.Dropout(0.3)  # 防止过拟合

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.dropout(y)  # Dropout层
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return self.ReLU(y + x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=128, rates=[3, 6]):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.conv_cat(x_cat)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512)
        )
        self.aspp = ASPP(512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),  # 降维减少计算量
            nn.BatchNorm1d(64),  # 增加 BN 以稳定训练
            nn.ReLU(),  # 或者 nn.GELU()
            nn.Dropout(0.4),  # 适度降低 Dropout 避免丢失太多信息
            nn.Linear(64, 14)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    print(summary(model, (3, 224, 224)))
