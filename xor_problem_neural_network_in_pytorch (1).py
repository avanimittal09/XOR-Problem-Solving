# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1ojXAyt7H8U4UZsAgsxIhE65JMNF622M2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def reset_seeds():
    np.random.seed(42)
    torch.manual_seed(42)

reset_seeds()

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Initialize the model, loss function and optimizer
reset_seeds()  # Reset seeds before model initialization
model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Reduce the learning rate

# Training loop
num_epochs = 10000
J_history = []

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the loss for plotting
    J_history.append(loss.item())

    # Print loss occasionally
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_tensor).float().mean().item()

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Predicted classes:', predicted_classes.numpy().flatten())
print('Actual classes:', y_tensor.numpy().flatten())
print('Predictions:', predictions.numpy().flatten())

# Plot the convergence of the cost function
plt.figure(figsize=(8, 5))
plt.plot(J_history)
plt.title('Convergence of Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

"""Discuss the results!"""

