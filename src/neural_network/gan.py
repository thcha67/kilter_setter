import sys
sys.path.append(r'C:\Users\thoma\OneDrive\Documents\kilter\kilter_setter')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from scripts.kilter_utils import load_training_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(1)))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 157 * 161, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(1)))  # Add a channel dimension
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 157 * 161)  # Flatten for the fully connected layer
        return torch.sigmoid(self.fc(x))

def train_gan(gen, disc, inputs, targets, lr=0.0002, num_epochs=10, save_every=None):
    gen = gen.to(device)
    disc = disc.to(device)

    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
    optimizer_disc = optim.Adam(disc.parameters(), lr=lr)

    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Train discriminator
            optimizer_disc.zero_grad()
            real_outputs = disc(targets.unsqueeze(1))  # Add a channel dimension
            real_loss = criterion(real_outputs, torch.ones_like(real_outputs))
            fake_inputs = gen(inputs)
            fake_outputs = disc(fake_inputs.detach().unsqueeze(1))  # Add a channel dimension
            fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            optimizer_disc.step()

            # Train generator
            optimizer_gen.zero_grad()
            fake_outputs = disc(fake_inputs.unsqueeze(1))  # Add a channel dimension
            gen_loss = criterion(fake_outputs, torch.ones_like(fake_outputs))
            gen_loss.backward()
            optimizer_gen.step()
        # Print and log metrics for monitoring convergence
        if save_every is not None and epoch % save_every == 0:
            torch.save(gen.state_dict(), f'models/gan_generator_{epoch}.pt')
            torch.save(disc.state_dict(), f'models/gan_discriminator_{epoch}.pt')
        print(f'Epoch {epoch} - Generator Loss: {gen_loss.item()} - Discriminator Loss: {disc_loss.item()} - Real Loss: {real_loss.item()} - Fake Loss: {fake_loss.item()}')

    return gen, disc

if __name__ == '__main__':
    load_training_data()
    # gen = Generator().to(device)
    # disc = Discriminator().to(device)

    # inputs, targets = load_training_data(device=device)

    # gen, disc = train_gan(gen, disc, inputs, targets, save_every=5)

    # torch.save(gen.state_dict(), 'models/gan_generator.pt')
    # torch.save(disc.state_dict(), 'models/gan_discriminator.pt')
