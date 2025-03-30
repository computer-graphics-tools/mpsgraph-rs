//! Sugar API Example
//!
//! This example demonstrates usage of the functional tensor operations
//! provided by the sugar_api feature.
//!
//! The API includes:
//! - Operator overloading (+, -, *, /, unary -)
//! - Functional tensor operations (square, sqrt, abs, etc.)
//! - ML activation functions (relu, silu, gelu)
//! - Tensor creation utilities
//!
//! Run with: `cargo run --example sugar_api_example --features="sugar_api"`

use mpsgraph::prelude::*;

fn main() {
    // Create graph and input tensors
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("b"));

    println!("Demonstrating sugar API operations with tensors:");
    println!("Shape = [2, 3]");

    // 1. Basic arithmetic with references
    let _sum = &a + &b;
    let _diff = &a - &b;
    let _product = &a * &b;
    let _division = &a / &b;
    let _negated = -&a;

    // 2. Functional operations
    let _squared = square(&a, None);
    let _sqrt_a = sqrt(&a, None);
    let _abs_diff = abs(&(&a - &b), None);

    // 3. Function composition
    let _complex_expr = abs(&sqrt(&(&a + &b), None), None);

    // 4. Activation functions
    let _silu_a = silu(&a, None);
    let _gelu_a = gelu(&a, None);

    // 5. Tensor creation
    let _zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    let _ones = graph.ones(&[2, 3], MPSDataType::Float32);
    let _random = graph.create_random_uniform(0.0, 1.0, &[2, 2], MPSDataType::Float32);

    // 6. Constants and operations with constants
    let half = graph.constant_scalar(0.5, MPSDataType::Float32);
    let _scaled = &a * &half;

    // 7. Additional operations
    let _exp_a = exp(&a, None);
    let _log_a = log(&a, None);

    // 8. Operations with multiple inputs
    let min_val = graph.constant_scalar(0.0, MPSDataType::Float32);
    let max_val = graph.constant_scalar(1.0, MPSDataType::Float32);
    let _clipped_a = clip(&a, &min_val, &max_val, None);

    // 9. Creating sequence data
    let _sequence = graph.arange(5.0, 5, MPSDataType::Float32);

    println!("\nSuccessfully created operations using functional style API:");
    println!("- Basic arithmetic: &a + &b, &a - &b, &a * &b, &a / &b, -&a");
    println!("- Simple operations: square(&a), sqrt(&a), abs(&tensor)");
    println!("- Complex expressions: abs(&sqrt(&(&a + &b)))");
    println!("- Activation functions: silu(&a), gelu(&a)");
    println!("- Math operations: exp(&a), log(&a), pow(&a, &exponent)");
    println!("- Tensor creation: zeros, ones, random_uniform");
    println!("- Sequence generation: arange");
    println!("- Range operations: clip(&a, &min, &max)");
}
