import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hn

    def initHidden(self, a=None):
        a = a or self.batch_size
        return torch.zeros(1, a, self.hidden_size)

def load_and_process_data(file_path, max_samples=1000):
    with open(file_path, "rb") as f:
        sequences = pickle.load(f)[:max_samples]

    next_sequences = [torch.tensor(seq[1:]) for seq in sequences]
    sequences = [torch.tensor(seq[:-1]) for seq in sequences]

    max_len = max([len(seq) for seq in sequences])

    for i in range(len(sequences)):
        pad_len = max_len - len(sequences[i])
        sequences[i] = torch.cat((sequences[i], torch.zeros(pad_len, 3)))
        next_sequences[i] = torch.cat((next_sequences[i], torch.zeros(pad_len, 3)))
    dataset = TensorDataset(torch.stack(sequences), torch.stack(next_sequences))
    return dataset

n_hidden = 512
batch_size = 50
rnn = RNN(input_size=3, hidden_size=n_hidden, output_size=3, batch_size=batch_size).to(device)

num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.05)
scheduler = StepLR(optimizer, step_size=50, gamma=0.2)

dataset = load_and_process_data("data/sequences.pkl", max_samples=5000)

dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    rnn.train()
    avg_loss = 0.0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        h0 = rnn.initHidden().to(device)  # Initialize hidden state for each batch

        optimizer.zero_grad()
        output, h0 = rnn(x, h0)
        loss = criterion(output.permute(0, 2, 1), y.permute(0, 2, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
        optimizer.step()
        avg_loss += loss.item()
        h0 = h0.detach()

    avg_loss /= len(dl)
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} : average loss = {avg_loss:.4f}")

torch.save(rnn.state_dict(), "data/rnn.pt")

with torch.no_grad():
    rnn.eval()
    h0 = rnn.initHidden(1).to(device)
    x = torch.tensor([[[88.0, 32.0, 1.0]]]).to(device)
    output, h0 = rnn(x, h0)
    print(output)
    