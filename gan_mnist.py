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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    # input tensor is of shape 32x1x28x28, the first line of forward transforms it to
    # -> 32x784
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 784)
        return self.model(x)


discriminator = Discriminator()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = output.view(output.size(0), 1, 28, 28)
        return output


generator = Generator().to(device)

# add training parameters
lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr,)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr,)

for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # training data for discriminator
        real_samples = real_samples.to(device)
        real_sample_labels = torch.ones((batch_size, 1)).to(device)

        latent_space_samples = torch.randn((batch_size, 100)).to(device)

        generated_samples = generator(latent_space_samples)
        generated_sample_labels = torch.zeros(batch_size, 1)

        all_samples = torch.cat((real_samples, generated_samples), dim=0,)
        all_sample_labels = torch.cat((real_sample_labels, generated_sample_labels), dim=0)

        # discriminator training
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_sample_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # data for training generator
        latent_space_samples = torch.randn((batch_size, 100)).to(device)

        # generator training
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_sample_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # display loss
        if n == batch_size-1:
            print(f"Epoch: {epoch} Loss Discriminator: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss Generator: {loss_generator}")

latent_space_samples = torch.randn((batch_size, 100)).to(device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
plt.show()
