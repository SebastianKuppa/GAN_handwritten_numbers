import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(111)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)

batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True)

real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
plt.show()
