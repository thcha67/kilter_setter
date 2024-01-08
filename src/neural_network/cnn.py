import sys
sys.path.append(r'C:\Users\thoma\OneDrive\Documents\kilter\kilter_setter')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 157 * 161)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(lr, num_epochs, overwrite=False):
    training_data = torch.tensor(np.load('data/training_data_flattened_uint8.npy'), dtype=torch.float32)

    input_data = training_data[:, -2:]
    target_hold_matrices = training_data[:, :-2]

    input_data = input_data.to(device)
    target_hold_matrices = target_hold_matrices.to(device)

    model = NeuralNetwork().to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert target_hold_matrices to one-hot encoded format for binary cross-entropy loss
    target_hold_matrices_one_hot = (target_hold_matrices > 0).float()

    # Create DataLoader for training
    dataset = TensorDataset(input_data, target_hold_matrices_one_hot)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss/len(val_loader)}')

    # Save model
    save_path = 'models/trained_model_cnn3_v1.pth'
    if not overwrite:
        while save_path in os.listdir('models'):
            i = 0
            save_path += str(i)
            i += 1
    
    torch.save(model.state_dict(), 'models/trained_model_cnn3_v1.pth')

if __name__ == '__main__':
    #train(lr=0.0001, num_epochs=100, overwrite=True)

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load('models/trained_model_cnn3_v1.pth'))
    model.eval()

    angle = 40
    difficulty = 15
    input_features = torch.tensor([angle, difficulty], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_features)
        output = output.view(157, 161)
        output = output.cpu().numpy()


    generated_hold_matrix = output

    from scripts.kilter_utils import plot_matrix

    plot_matrix(generated_hold_matrix)
