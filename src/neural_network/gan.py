import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 157*161)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(157*161, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def train_gan(gen, disc, inputs, targets, lr=0.0002, num_epochs=10):
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
            real_outputs = disc(targets)
            real_loss = criterion(real_outputs, torch.ones_like(real_outputs))
            fake_inputs = gen(inputs)
            fake_outputs = disc(fake_inputs.detach())
            fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            optimizer_disc.step()

            # Train generator
            optimizer_gen.zero_grad()
            fake_outputs = disc(fake_inputs)
            gen_loss = criterion(fake_outputs, torch.ones_like(fake_outputs))
            gen_loss.backward()
            optimizer_gen.step()

        # Print and log metrics for monitoring convergence
        print(f'Epoch {epoch} - Generator Loss: {gen_loss.item()} - Discriminator Loss: {disc_loss.item()} - Real Loss: {real_loss.item()} - Fake Loss: {fake_loss.item()}')

    return gen, disc

def load_training_data(path):
    training_data = torch.tensor(np.load(path), dtype=torch.float32).to(device)
    inputs = training_data[:, :2]
    targets = training_data[:, 2:] / 15
    return inputs, targets

if __name__ == '__main__':
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    inputs, targets = load_training_data('data/training_data_flat_uint8.npy')

    gen, disc = train_gan(gen, disc, inputs, targets)

    torch.save(gen.state_dict(), 'models/gan_generator.pt')
    torch.save(disc.state_dict(), 'models/gan_discriminator.pt')
