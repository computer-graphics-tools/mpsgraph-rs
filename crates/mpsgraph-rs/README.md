# mpsgraph-rs

A Rust wrapper for Apple's MetalPerformanceShadersGraph (MPSGraph) API, enabling high-performance, GPU-accelerated machine learning and numerical computing on Apple platforms.

## Features

- **Complete API Coverage**: Comprehensive bindings to MetalPerformanceShadersGraph
- **Safe Memory Management**: Proper Rust ownership semantics with automatic resource cleanup
- **Efficient Graph Execution**: Synchronous and asynchronous execution options
- **Type Safety**: Strong typing with Rust's type system
- **Tensor Operations**: Full suite of tensor operations for numerical computing and machine learning

## Requirements

- macOS, iOS, tvOS or other Apple platform with Metal support
- Rust 1.58+

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mpsgraph = "0.1.0"
```

## Example

```rust
use mpsgraph::{Graph, MPSShapeDescriptor, MPSDataType};
use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;

fn main() {
    // Get the Metal device
    let device = Device::system_default().expect("No Metal device found");
    
    // Create a graph
    let graph = Graph::new().expect("Failed to create graph");
    
    // Create input tensors
    let shape = MPSShapeDescriptor::new(vec![2, 3], MPSDataType::Float32);
    let x = graph.placeholder(&shape, Some("x"));
    let y = graph.placeholder(&shape, Some("y"));
    
    // Define the computation: z = x + y
    let z = graph.add(&x, &y, Some("z"));
    
    // Create input data
    let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let y_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3 matrix
    
    // Create Metal buffers
    let buffer_size = (6 * std::mem::size_of::<f32>()) as u64;
    let x_buffer = device.new_buffer_with_data(
        x_data.as_ptr() as *const _, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );
    let y_buffer = device.new_buffer_with_data(
        y_data.as_ptr() as *const _, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );
    
    // Create feed dictionary
    let mut feed_dict = HashMap::new();
    feed_dict.insert(&x, x_buffer.deref());
    feed_dict.insert(&y, y_buffer.deref());
    
    // Run the graph
    let results = graph.run(&device, feed_dict, &[&z]);
    
    // Process results
    unsafe {
        let result_ptr = results[0].contents() as *const f32;
        let result_values = std::slice::from_raw_parts(result_ptr, 6);
        println!("Result: {:?}", result_values);
        // Outputs: [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    }
}
```

## Additional Features

- Matrix multiplication and other linear algebra operations
- Activation functions (ReLU, sigmoid, tanh, etc.)
- Reduction operations (sum, mean, max, min)
- Tensor reshaping and transposition
- Graph compilation for repeated execution

## Advanced Example: Neural Network with MPSGraph

```rust
use mpsgraph::{Graph, MPSShapeDescriptor, MPSDataType};
use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;

fn main() {
    // Get the system default Metal device
    let device = Device::system_default().expect("No Metal device found");
    
    // Create a graph
    let graph = Graph::new().expect("Failed to create graph");
    
    // Define the neural network architecture
    // Input: 784 features (28x28 image flattened)
    // Hidden layer: 128 neurons with ReLU activation
    // Output: 10 classes with softmax activation
    
    // Input placeholder
    let input_shape = MPSShapeDescriptor::new(vec![1, 784], MPSDataType::Float32);
    let x = graph.placeholder(&input_shape, Some("input"));
    
    // First layer weights and biases
    let w1_shape = MPSShapeDescriptor::new(vec![784, 128], MPSDataType::Float32);
    let b1_shape = MPSShapeDescriptor::new(vec![1, 128], MPSDataType::Float32);
    
    // Create random weights (would normally be trained or loaded from a file)
    let mut w1_data = vec![0.0f32; 784 * 128];
    let mut b1_data = vec![0.0f32; 128];
    
    // Initialize with small random values (simplified)
    for i in 0..w1_data.len() {
        w1_data[i] = (i as f32 * 0.0001) - 0.05;
    }
    
    let w1 = graph.constant_with_data(&w1_data, &w1_shape, Some("w1"));
    let b1 = graph.constant_with_data(&b1_data, &b1_shape, Some("b1"));
    
    // First layer computation: h1 = ReLU(x · w1 + b1)
    let xw1 = graph.matmul(&x, &w1, Some("xw1"));
    let xw1_plus_b1 = graph.add(&xw1, &b1, Some("logits1"));
    let h1 = graph.relu(&xw1_plus_b1, Some("hidden1"));
    
    // Second layer (output layer)
    let w2_shape = MPSShapeDescriptor::new(vec![128, 10], MPSDataType::Float32);
    let b2_shape = MPSShapeDescriptor::new(vec![1, 10], MPSDataType::Float32);
    
    let mut w2_data = vec![0.0f32; 128 * 10];
    let mut b2_data = vec![0.0f32; 10];
    
    // Initialize with small random values (simplified)
    for i in 0..w2_data.len() {
        w2_data[i] = (i as f32 * 0.001) - 0.05;
    }
    
    let w2 = graph.constant_with_data(&w2_data, &w2_shape, Some("w2"));
    let b2 = graph.constant_with_data(&b2_data, &b2_shape, Some("b2"));
    
    // Output layer computation: y = softmax(h1 · w2 + b2)
    let h1w2 = graph.matmul(&h1, &w2, Some("h1w2"));
    let logits = graph.add(&h1w2, &b2, Some("logits"));
    let probs = graph.softmax(&logits, 1, Some("probabilities"));
    
    // Create a sample input (a simplified image)
    let mut input_data = vec![0.0f32; 784];
    for i in 0..784 {
        // Create a simple pattern
        input_data[i] = if (i / 28 + i % 28) % 2 == 0 { 0.9 } else { 0.1 };
    }
    
    // Create input buffer
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const _,
        (784 * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared
    );
    
    // Create feed dictionary
    let mut feed_dict = HashMap::new();
    feed_dict.insert(&x, input_buffer.deref());
    
    // Run the graph
    let results = graph.run(&device, feed_dict, &[&probs]);
    assert_eq!(results.len(), 1);
    
    // Process and print the results (class probabilities)
    unsafe {
        let probs_ptr = results[0].contents() as *const f32;
        let probabilities = std::slice::from_raw_parts(probs_ptr, 10);
        
        println!("Class probabilities:");
        for (i, &prob) in probabilities.iter().enumerate() {
            println!("  Class {}: {:.6}", i, prob);
        }
        
        // Find the predicted class (highest probability)
        let mut max_idx = 0;
        let mut max_prob = probabilities[0];
        for i in 1..10 {
            if probabilities[i] > max_prob {
                max_idx = i;
                max_prob = probabilities[i];
            }
        }
        
        println!("Predicted class: {} (probability: {:.6})", max_idx, max_prob);
    }
}

## License

Licensed under the MIT License.
