import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# adding a random seed helps to replicate results identically on different machines
# The seed number represents the random seed used to initialize the random number generator,
# which is used to initialize the neural network’s weights
torch.manual_seed(123)

train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2*math.pi*torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

