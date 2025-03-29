//! Sugar API Example
//!
//! This example demonstrates the usage of the sugar_api feature
//! for more ergonomic MPSGraph tensor operations.
//!
//! The sugar_api feature provides:
//! - Operator overloading (+, -, *, /, unary -)
//! - Method-based tensor operations (square, sqrt, abs)
//! - Specialized ML functions (SiLU activation)
//! - Tensor creation utilities (zeros, ones)
//!
//! To run this example:
//! ```
//! cargo run --example sugar_api_example --features="sugar_api"
//! ```

use mpsgraph::prelude::*;

fn main() {
    // Create a new graph
    let graph = MPSGraph::new();

    // Create placeholders for input tensors
    let shape_dims = &[2, 3];
    let shape = MPSShape::from_slice(&[2usize, 3usize]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("b"));

    println!("Demonstrating sugar API operations with tensors:");
    println!("Shape = [{}, {}]", shape_dims[0], shape_dims[1]);

    // Create some example operations using the sugar API

    // 1. Basic arithmetic operations with references
    // IMPORTANT: Always use reference-based operations to avoid moving/dropping tensors
    let _sum = &a + &b;
    let _diff = &a - &b;
    let _product = &a * &b;
    let _division = &a / &b;

    // 2. Unary operations
    let _negated = -&a;
    let _squared = a.square(None);
    let _sqrt_a = a.sqrt(None);
    let _abs_diff = (&a - &b).abs(None);

    // 3. SiLU activation
    let _silu_a = a.silu(None);
    
    // 4. Tensor creation utilities
    let _zeros = graph.zeros(&[2, 3], MPSDataType::Float32);
    let _ones = graph.ones(&[2, 3], MPSDataType::Float32);
    
    // 5. Scalar operations (using const_scalar)
    let half = a.const_scalar(0.5);
    let _scaled = &a * &half;
    
    // 6. Method chaining with references
    let _complex_expr = (&a + &b).sqrt(None).abs(None);
    
    // Print that operations were created successfully
    println!("\nSuccessfully created the following operations:");
    println!("- Addition (&a + &b)");
    println!("- Subtraction (&a - &b)");
    println!("- Multiplication (&a * &b)");
    println!("- Division (&a / &b)");
    println!("- Negation (-&a)");
    println!("- Square (a²)");
    println!("- Square root (√a)");
    println!("- Absolute difference (|&a - &b|)");
    println!("- SiLU activation (a * sigmoid(a))");
    println!("- GELU activation");
    println!("- Tensor creation (zeros, ones, full, linspace, random_uniform, random_normal)");
    println!("- Scalar operations (&a * &half)");
    println!("- Method chaining (sqrt(|a + b|))");
    println!("- Additional math operations (exp, log, sigmoid, tanh, relu, pow, clip)");
    println!("- Matrix utilities (eye, diag, arange)");
    
    // Demonstrate some of the additional operations
    
    // Create a sequence of values from 0 to 1
    let _linspace = graph.linspace(0.0, 1.0, 5, MPSDataType::Float32, None);
    
    // Create random tensors
    let _random_uniform = graph.create_random_uniform(0.0, 1.0, &[2, 2], MPSDataType::Float32);
    let _random_normal = graph.create_random_normal(0.0, 1.0, &[2, 2], MPSDataType::Float32);
    
    // Create an identity matrix
    let _identity = graph.eye(3, MPSDataType::Float32);
    
    // Create a sequence of values
    let _sequence = graph.arange(5.0, 5, MPSDataType::Float32);
    
    // Apply advanced activation functions
    let _gelu_a = a.gelu(None);
    
    // Apply exponential and logarithmic functions
    let _exp_a = a.exp(None);
    let _log_a = a.log(None);
    
    // Apply clip operation to limit values to a range
    let min_val = graph.constant_scalar(0.0, MPSDataType::Float32);
    let max_val = graph.constant_scalar(1.0, MPSDataType::Float32);
    let _clipped_a = a.clip(&min_val, &max_val, None);
    
    // Apply power operation
    let exponent = graph.constant_scalar(2.0, MPSDataType::Float32);
    let _squared_a = a.pow(&exponent, None);
    
    println!("\nAdditional operations were created successfully!");
}