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


def train_model(model, inputs, targets, lr=0.001, num_epochs=10):
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def normalize_matrix(matrix):
    return matrix / matrix.max()

def load_training_data(path):
    training_data = torch.tensor(np.load(path), dtype=torch.float32).to(device)
    inputs = training_data[:, :2]  # (num_samples, 2)
    targets = training_data[:, 2:]  #  (num_samples, 157*161)
    targets = normalize_matrix(targets)
    return inputs, targets

if __name__ == '__main__':
    model = Generator().to(device)

    inputs, targets = load_training_data('data/training_data_flat_uint8.npy')

    model = train_model(model, inputs, targets)

    torch.save(model.state_dict(), 'models/generator.pt')
