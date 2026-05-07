import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------------
# Define a Simple Vanilla RNN Model
# -------------------------------
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Initialize hidden state (num_layers=1)
        out, _ = self.rnn_cell(x, h0)  # RNN forward pass
        out = self.fc(out[:, -1, :])  # Take the last time step's output for classification
        return out

# -------------------------------
# Generate Toy Dataset (Sequential Classification)
# -------------------------------
def generate_data(num_samples=1000, seq_length=10):
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.randn(seq_length, 1)  # Random sequence
        label = 1 if np.mean(seq) > 0 else 0  # Classify based on mean value
        X.append(seq)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# Prepare dataset
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# -------------------------------
# Training the RNN Model
# -------------------------------
input_size = 1
hidden_size = 16
output_size = 2  # Binary classification
num_epochs = 10
batch_size = 32
learning_rate = 0.01

# Initialize model, loss, and optimizer
model = VanillaRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -------------------------------
# Evaluating the Model
# -------------------------------
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")