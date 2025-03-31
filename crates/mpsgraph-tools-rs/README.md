# mpsgraph-tools

A high-level Rust API for working with Apple's Metal Performance Shaders Graph (MPSGraph) framework, providing ergonomic tensor operations with operator overloading and a functional programming style.

## Features

- **Tensor Operations API**: Ergonomic, functional-style tensor operations with operator overloading
- **Utility Functions**: Convenience methods for common tensor operations
- **Tensor Creation Helpers**: Easy creation of tensors with different initialization patterns
- **Extension Traits**: Convenient methods added to core MPSGraph types

## Requirements

- macOS, iOS, or other Apple platform that supports Metal
- Rust 1.85 or newer

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
mpsgraph = "0.1.0"
mpsgraph-tools = "0.1.0"
```

## Example

```rust
use mpsgraph_tools::prelude::*;

fn main() {
    // Create graph and input tensors
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, MPSDataType::Float32, Some("b"));

    // Use operator overloading with references
    let sum = &a + &b;
    let diff = &a - &b;
    
    // Use functional operations
    let squared = square(&a, None);
    let abs_diff = abs(&(&a - &b), None);
    
    // Compose operations
    let complex_expr = abs(&sqrt(&(&a + &b), None), None);
    
    // Create specialized tensors
    let zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    let ones = graph.ones(&[2, 3], MPSDataType::Float32);
    let full = graph.full(5.0f32, &[2, 3], MPSDataType::Float32);
    let random = graph.create_random_uniform(0.0, 1.0, &[2, 2], MPSDataType::Float32);
    
    // Method chaining
    let result = a.square(None)
                  .sigmoid(None)
                  .clip(&graph.const_tensor(0.1f32), &graph.const_tensor(0.9f32), None);
}
```

## Key Components

### Tensor Wrapper

The `Tensor` struct wraps an `MPSGraphTensor` and provides:

- Operator overloading (`+`, `-`, `*`, `/`, etc.)
- Method-based operations (`.square()`, `.sigmoid()`, etc.)
- Easy conversion to and from `MPSGraphTensor`

### GraphExt Trait

The `GraphExt` trait extends `MPSGraph` with additional methods:

- `zeros()`, `ones()`, `full()` - Create tensors with specific values
- `placeholder_tensor()` - Create placeholder tensors with automatic wrapping
- `create_random_uniform()`, `create_random_normal()` - Create random tensors

## Available Operations

- **Arithmetic**: Add, subtract, multiply, divide, negate
- **Unary Operations**: Square, sqrt, abs, exp, log
- **Activation Functions**: Sigmoid, tanh, relu, silu, gelu
- **Binary Operations**: Power, clip
- **Tensor Creation**: Zeros, ones, full, random tensors

## Advanced Usage Example: Neural Network Layer

```rust
use mpsgraph_tools::prelude::*;

fn main() {
    // Create graph
    let graph = MPSGraph::new();
    
    // Input: batch_size x input_features
    let input_shape = MPSShape::from_slice(&[32, 784]);
    let input = graph.placeholder_tensor(&input_shape, MPSDataType::Float32, Some("input"));
    
    // Weights: input_features x output_features
    let weights_shape = MPSShape::from_slice(&[784, 128]);
    let weights = graph.create_random_normal(0.0, 0.01, &[784, 128], MPSDataType::Float32);
    
    // Bias: 1 x output_features
    let bias_shape = MPSShape::from_slice(&[1, 128]);
    let bias = graph.zeros(&[1, 128], MPSDataType::Float32);
    
    // Forward pass: y = relu(x · W + b)
    let xw = graph.matmul_tensor(&input, &weights, None);
    let logits = &xw + &bias;
    let hidden = relu(&logits, None);
    
    // Add a second layer
    let weights2 = graph.create_random_normal(0.0, 0.01, &[128, 10], MPSDataType::Float32);
    let bias2 = graph.zeros(&[1, 10], MPSDataType::Float32);
    
    // Output layer: softmax(hidden · W2 + b2)
    let xw2 = graph.matmul_tensor(&hidden, &weights2, None);
    let logits2 = &xw2 + &bias2;
    let output = logits2.softmax(1, None);
    
    // The network is now defined and ready for execution
    println!("Neural network defined with input shape: {:?}", input.inner().shape().dimensions());
    println!("Output shape: {:?}", output.inner().shape().dimensions());
}
```

## License

Licensed under the MIT License.