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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


discriminator = Discriminator()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


generator = Generator()

# init training params
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# training loop
for epoch in range(num_epochs):
    for n, (real_examples, _) in enumerate(train_loader):
        # init data examples and labels for training
        real_sample_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_sample_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_examples, generated_samples))
        all_sample_labels = torch.cat((real_sample_labels, generated_sample_labels))

        # train the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_sample_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # training data for generator
        latent_space_samples = torch.randn((batch_size, 2))

        # training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_sample_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

