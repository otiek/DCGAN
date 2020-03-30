import torch
import torch.nn as nn


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(ConvTranspose, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(Conv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvTranspose(10, 128),
            ConvTranspose(128, 64),
            ConvTranspose(64, 32),
            ConvTranspose(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv(1, 32),
            Conv(32, 64),
            Conv(64, 128),
            Conv(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    #model = nn.ConvTranspose2d(10, 128, kernel_size=2, stride=2)
    #model = Conv(1, 32)
    model = Discriminator()
    x = torch.randn(5, 1, 64, 64)
    y = model(x)
    print(y.size())