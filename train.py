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
sample_img = mnist[0][0]
size = sample_img.squeeze().size()
print('Total samples: {}'.format(len(mnist)))
print('Image size: {}x{}'.format(size[0], size[1]))