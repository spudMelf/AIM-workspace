import torch
import numpy as np
import math
import torch.nn as nn
from heart_data_set_loader import HeartDataset
from torch.utils.data import Dataset, DataLoader, random_split

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 20
hidden_size = 100
num_classes = 1
num_epochs = 100
batch_size = 100
learning_rate = .0001

# dataset
dataset = HeartDataset()
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# neural network

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
print("Beginning training loop")
n_total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (entries, labels) in enumerate(train_dataloader):
        # reshape to column vector
        entries = entries.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(entries)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'epoch: {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for entries, labels in test_dataloader:
        entries = entries.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(entries)

        # Apply sigmoid activation to convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities > 0.5).float()

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = n_correct / n_samples
    print(f'accuracy = {acc}')

