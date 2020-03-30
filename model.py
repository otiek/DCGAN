import torch
import torch.nn as nn


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvTranspose, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvTranspose(100, 128, stride=2, padding=1),
            ConvTranspose(128, 64, kernel_size=3, stride=2, padding=1),
            ConvTranspose(64, 32),
            ConvTranspose(32, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    pass