# XOR Non-Linear Classifier

A simple neural network built with PyTorch to solve the XOR logic problem.

## Overview
A basic single-layer perceptron can't solve XOR because the data isn't linearly separable. This project uses a small Multi-Layer Perceptron (MLP) with a hidden layer and non-linear activation functions to get around that limitation.

## What's Inside
- **Model (`XORNet`):** Built using PyTorch's `nn.Module` with:
  - A hidden linear layer using a `Tanh` activation function.
  - An output linear layer using a `Sigmoid` activation function.
- **Training:** Uses Binary Cross-Entropy Loss (`nn.BCELoss`) and Stochastic Gradient Descent (`torch.optim.SGD`).
- **Tracking:** Sets random seeds for consistent results, logs the loss history, and plots the convergence using `matplotlib`.
- **Results:** Trains for 10,000 epochs and reaches 100% accuracy on the XOR truth table.

## Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib
