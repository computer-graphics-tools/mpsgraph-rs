# MPSGraph Tools

High-level utilities and ergonomic APIs for working with Apple's Metal Performance Shaders Graph (MPSGraph) framework in Rust.

## Features

- **Tensor Operations API**: Ergonomic, functional-style tensor operations with operator overloading
- **Utility Functions**: Convenience methods for common tensor operations
- **Tensor Creation Helpers**: Easy creation of tensors with different initialization patterns

## Usage

Add the following to your `Cargo.toml`:

```toml
[dependencies]
mpsgraph = { path = "../mpsgraph-rs" }
mpsgraph-tools = { path = "../mpsgraph-tools-rs" }
```

Then in your code:

```rust
use mpsgraph_tools::prelude::*;

fn main() {
    // Create graph and input tensors
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, None);
    let b = graph.placeholder(&shape, MPSDataType::Float32, None);

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
    let random = graph.create_random_uniform(0.0, 1.0, &[2, 2], MPSDataType::Float32);
}
```

See the `examples` directory for more usage examples.

## License

This project is licensed under the MIT License - see the LICENSE file for details.