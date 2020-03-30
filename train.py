import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


mnist = MNIST('../Dataset', transform=transforms.ToTensor(), download=True)
print(type(mnist))
print(len(mnist))
data = mnist[0]
img = data[0]
num = data[1]
print(img.size())
print(num)