use std::collections::HashMap;

// Import basic components
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;

// Import specific components needed for stencil operations
use mpsgraph::stencil_ops::{MPSGraphStencilOpDescriptor, MPSGraphReductionMode};
use mpsgraph::sample_grid_ops::MPSGraphPaddingMode;
use mpsgraph::convolution_transpose_ops::PaddingStyle;

fn main() {
    println!("Testing all operations API functionality");
    
    // Test representative operations from each category
    test_arithmetic_ops();
    test_activation_ops();
    test_reduction_ops();
    test_linear_algebra_ops();
    test_stencil_ops();
    test_memory_ops();
    test_cumulative_ops();
    
    println!("All tests completed successfully");
}

fn test_arithmetic_ops() {
    println!("\nTesting arithmetic operations...");
    
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2, 2]);
    
    // Create input tensors
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("b"));
    
    // Create operations
    let add = graph.add(&a, &b, Some("add"));
    let sub = graph.subtract(&a, &b, Some("sub"));
    let mul = graph.multiply(&a, &b, Some("mul"));
    let div = graph.divide(&a, &b, Some("div"));
    
    // Create input data
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];
    
    let tensor_data_a = MPSGraphTensorData::from_bytes(&data_a, &shape, MPSDataType::Float32);
    let tensor_data_b = MPSGraphTensorData::from_bytes(&data_b, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&a, tensor_data_a);
    feeds.insert(&b, tensor_data_b);
    
    // Run the graph
    let _results = graph.run(feeds, &[&add, &sub, &mul, &div]);
    
    println!("Arithmetic operations test completed");
}

fn test_activation_ops() {
    println!("\nTesting activation operations...");
    
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[5]);
    
    // Create input tensor
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    
    // Create operations
    let relu = graph.relu(&input, Some("relu"));
    let sigmoid = graph.sigmoid(&input, Some("sigmoid"));
    let tanh = graph.tanh(&input, Some("tanh"));
    
    // Create input data
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&input, tensor_data);
    
    // Run the graph
    let _results = graph.run(feeds, &[&relu, &sigmoid, &tanh]);
    
    println!("Activation operations test completed");
}

fn test_reduction_ops() {
    println!("\nTesting reduction operations...");
    
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2, 3]);
    
    // Create input tensor
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    
    // Create operations
    let sum_cols = graph.reduce_sum(&input, &[1], Some("sum_cols"));
    let mean_cols = graph.reduce_mean(&input, &[1], Some("mean_cols"));
    let max_cols = graph.reduce_max(&input, &[1], Some("max_cols"));
    let min_cols = graph.reduce_min(&input, &[1], Some("min_cols"));
    
    // Create input data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&input, tensor_data);
    
    // Run the graph
    let _results = graph.run(feeds, &[&sum_cols, &mean_cols, &max_cols, &min_cols]);
    
    println!("Reduction operations test completed");
}

fn test_linear_algebra_ops() {
    println!("\nTesting linear algebra operations...");
    
    let graph = MPSGraph::new();
    
    // Test matrix multiplication
    let shape_a = MPSShape::from_slice(&[2, 3]);
    let shape_b = MPSShape::from_slice(&[3, 2]);
    
    let a = graph.placeholder(&shape_a, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape_b, MPSDataType::Float32, Some("b"));
    
    let matmul = graph.matmul(&a, &b, Some("matmul"));
    
    // Create input data
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    
    let tensor_data_a = MPSGraphTensorData::from_bytes(&data_a, &shape_a, MPSDataType::Float32);
    let tensor_data_b = MPSGraphTensorData::from_bytes(&data_b, &shape_b, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&a, tensor_data_a);
    feeds.insert(&b, tensor_data_b);
    
    let _results = graph.run(feeds, &[&matmul]);
    
    // Test matrix inverse
    let shape = MPSShape::from_slice(&[2, 2]);
    let matrix = graph.placeholder(&shape, MPSDataType::Float32, Some("matrix"));
    let inverse = graph.matrix_inverse(&matrix, Some("inverse"));
    
    // Create input data
    let data = vec![4.0f32, 7.0, 2.0, 6.0];
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&matrix, tensor_data);
    
    let _results3 = graph.run(feeds.clone(), &[&inverse]);
    
    // Test band part
    let band_part = graph.band_part(&matrix, 0, 0, Some("band_part"));
    let _results2 = graph.run(feeds, &[&band_part]);
    
    println!("Linear algebra operations test completed");
}

fn test_stencil_ops() {
    println!("\nTesting stencil operations...");
    
    let graph = MPSGraph::new();
    
    // Define a simple 1x5x5x1 input (NHWC)
    let source_shape = MPSShape::from_slice(&[1, 5, 5, 1]);
    let source_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0
    ];
    let source = graph.placeholder(&source_shape, MPSDataType::Float32, None);
    
    // Define a 3x3x1x1 weights
    let weights_shape = MPSShape::from_slice(&[3, 3, 1, 1]);
    let weights_data = vec![
        1.0f32, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    ];
    let weights = graph.placeholder(&weights_shape, MPSDataType::Float32, None);
    
    // Create descriptor with sum reduction and explicit padding
    let offsets = MPSShape::from_slice(&[0, 0, 0, 0]);
    let strides = MPSShape::from_slice(&[1, 1, 1, 1]);
    let dilation_rates = MPSShape::from_slice(&[1, 1, 1, 1]);
    let explicit_padding = MPSShape::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    
    let descriptor = MPSGraphStencilOpDescriptor::with_all_params(
        MPSGraphReductionMode::Sum,
        &offsets,
        &strides,
        &dilation_rates,
        &explicit_padding,
        MPSGraphPaddingMode::Zero,
        PaddingStyle::Explicit,
        0.0
    );
    
    // Create the stencil operation
    let result = graph.stencil(
        &source,
        &weights,
        &descriptor,
        Some("stencil_op")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&source, MPSGraphTensorData::from_bytes(&source_data, &source_shape, MPSDataType::Float32));
    feeds.insert(&weights, MPSGraphTensorData::from_bytes(&weights_data, &weights_shape, MPSDataType::Float32));
    
    let _results = graph.run(feeds, &[&result]);
    
    println!("Stencil operations test completed");
}

fn test_memory_ops() {
    println!("\nTesting memory operations...");
    
    let graph = MPSGraph::new();
    
    // Create a variable tensor
    let shape = MPSShape::from_slice(&[2, 2]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let constant = graph.constant_from_bytes(&data, &shape, MPSDataType::Float32, None);
    let variable = graph.variable(&shape, MPSDataType::Float32, Some("variable"));
    
    // Test variable read
    let read = graph.read_variable(&variable, Some("read"));
    
    // Run the graph
    // Create an assign operation 
    let assign = graph.assign_variable(&variable, &constant, Some("assign"));
    
    // Run the graph with no feeds since we're using constants
    let _results = graph.run(HashMap::new(), &[&assign, &read]);
    
    println!("Memory operations test completed");
}

fn test_cumulative_ops() {
    println!("\nTesting cumulative operations...");
    
    let graph = MPSGraph::new();
    
    // Create a 2x3 tensor
    let shape = MPSShape::from_slice(&[2, 3]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    let tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    // Compute cumulative sum along columns (axis 0)
    let cum_sum_0 = graph.cumulative_sum(
        &tensor, 
        0,     // axis = 0 (along columns)
        false, // exclusive = false (inclusive)
        false, // reverse = false (forward)
        Some("cum_sum_0")
    );
    
    // Compute cumulative product along rows (axis 1)
    let cum_prod_1 = graph.cumulative_product(
        &tensor, 
        1,     // axis = 1 (along rows)
        false, // exclusive = false (inclusive)
        false, // reverse = false (forward)
        Some("cum_prod_1")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&tensor, tensor_data);
    
    let _results = graph.run(feeds, &[&cum_sum_0, &cum_prod_1]);
    
    println!("Cumulative operations test completed");
}