import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Define Generator and Discriminator architectures
class Generator(nn.Module):
    def __init__(self, input_size=2, output_size=40*37):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),   # Input size is 2, output size is 128
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),  # Increase the dimensionality
            nn.ReLU(),
            nn.Linear(512, output_size),  # Output size is 40 * 37 = 1480
            nn.Sigmoid()  # Sigmoid activation to get values between 0 and 1
        )

    def forward(self, input_params):
        x = self.fc(input_params)
        x = x.view(-1, 40, 37)  # Reshape to the desired output size
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size=40*37):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()  # Sigmoid activation to get a probability between 0 and 1
        )

    def forward(self, sample):
        sample = sample.view(-1, self.input_size)  # Flatten the input if it's a matrix
        prob_real_or_fake = self.fc(sample)
        return prob_real_or_fake

def train_model(inputs, targets, device):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)


    # Define loss functions and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.00007)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00007)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(inputs))
    val_size = len(inputs) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(TensorDataset(inputs, targets), [train_size, val_size])

    # Create DataLoader for training and validation
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training the GAN
    num_epochs = 200

# Inside the training loop
    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in train_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            discriminator.zero_grad()
            generator.zero_grad()
            
            # Real samples
            real_labels = torch.ones(batch_size, 1).to(device)
            real_outputs = discriminator(batch_targets)
            loss_real = criterion(real_outputs, real_labels)
            loss_d_real = -loss_real

            # Fake samples
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_samples = generator(batch_inputs)
            fake_outputs = discriminator(fake_samples.detach())
            loss_fake = criterion(fake_outputs, fake_labels)

            loss_d = loss_d_real + loss_fake

            loss_d.backward()
            optimizer_d.step()

            # Train the generator
            generator.zero_grad()
            fake_outputs = discriminator(fake_samples)
            loss_g = criterion(fake_outputs, real_labels)

            loss_g.backward()
            optimizer_g.step()

        # Validation (optional)
        with torch.no_grad():
            for val_batch_inputs, val_batch_targets in val_dataloader:
                val_batch_inputs = val_batch_inputs.to(device)
                val_batch_targets = val_batch_targets.to(device)

                # Evaluate discriminator on validation data
                val_real_outputs = discriminator(val_batch_targets)
                val_loss_real = criterion(val_real_outputs, torch.ones_like(val_real_outputs))
                val_fake_samples = generator(val_batch_inputs)
                val_fake_outputs = discriminator(val_fake_samples.detach())
                val_loss_fake = criterion(val_fake_outputs, torch.zeros_like(val_fake_outputs))
                val_loss_discriminator = val_loss_real + val_loss_fake

                # Evaluate generator on validation data
                val_fake_outputs_gen = discriminator(val_fake_samples)
                val_loss_generator = criterion(val_fake_outputs_gen, torch.ones_like(val_fake_outputs_gen))

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss D Real: {loss_d_real.item():.4f}, Loss D Fake: {loss_fake.item():.4f}, Loss G: {loss_g.item():.4f}')
        print(f'Validation Loss D: {val_loss_discriminator.item():.4f}, Validation Loss G: {val_loss_generator.item():.4f}')

        # if epoch % 5 == 0:
        #     torch.save(generator.state_dict(), f'models/gan_generator_{epoch}.pt')
        #     torch.save(discriminator.state_dict(), f'models/gan_discriminator_{epoch}.pt')
    torch.save(generator.state_dict(), f'models/gan_generator.pt')




if __name__ == '__main__':
    from scripts.kilter_utils import load_training_data, normalize_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs, targets = load_training_data(device=device, path_input='data/test_inputs.npy', path_target='data/test_targets.npy')

    inputs, targets = normalize_data(inputs, targets)

    train_model(inputs, targets, device=device)


