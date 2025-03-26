# MPSGraph-rs Examples

This directory contains example applications demonstrating how to use the MPSGraph-rs Rust bindings for Apple's Metal Performance Shaders Graph (MPSGraph) API.

## Available Examples

### 1. Matrix Multiplication (`matmul.rs`)

A simple example demonstrating matrix multiplication using MPSGraph. This example:
- Creates two matrices (2x3 and 3x2)
- Performs a matrix multiplication operation
- Prints the resulting 2x2 matrix

To run:
```
cargo run --example matmul
```

### 2. Neural Network Inference (`neural_network.rs`)

A more complex example implementing a simple feedforward neural network with:
- 784 input neurons (representing a flattened 28x28 image)
- 128 hidden neurons with ReLU activation
- 10 output neurons with softmax activation

This example demonstrates how to:
- Define a multi-layer neural network architecture
- Create weights and biases as constants
- Perform forward inference
- Process the output probabilities

To run:
```
cargo run --example neural_network
```

## Requirements

- macOS with Metal support
- Xcode with Metal development tools installed
- Rust toolchain

## Note

These examples use randomly initialized weights and dummy input data. In a real application, you would:
- Load pre-trained weights from a file
- Process real input data (like images)
- Perhaps run the model on a real task like image classification

The examples are meant to demonstrate the API usage patterns, not to achieve any meaningful ML task.