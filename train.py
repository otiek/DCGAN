import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# resize MNIST images (28, 28) to (64, 64)
tr = transforms.Compose([
    transforms.Resize((64, 64), interpolation=3),
    transforms.ToTensor()
])
#download and load MNIST data
mnist = MNIST('../Dataset', transform=tr, download=True)
print(type(mnist))
print(len(mnist))
data = mnist[0]
img = data[0]
num = data[1]
print(img.size())
print(num)