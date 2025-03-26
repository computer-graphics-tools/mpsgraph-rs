use metal::Device;
use std::collections::HashMap;

// Import the necessary components from the mpsgraph crate
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;
use mpsgraph::activation_ops::*;
use mpsgraph::arithmetic_ops::*;
use mpsgraph::reduction_ops::*;
use mpsgraph::tensor_shape_ops::*;
use mpsgraph::matrix_ops::*;
use mpsgraph::operation::*;
use mpsgraph::executable::*;

fn main() {
    // Check if a Metal device is available
    if let Some(device) = Device::system_default() {
        println!("Using Metal device: {}", device.name());
        run_metal_tests();
    } else {
        println!("No Metal device found.");
    }
}

fn run_metal_tests() {
    println!("Running MPSGraph tests on real Metal hardware");
    
    // Test basic arithmetic operations
    test_basic_arithmetic();
    
    // Test activation functions
    test_activation_functions();
    
    // Test reduction operations
    test_reduction_operations();
    
    // Test matrix operations
    test_matrix_multiplication();
    
    // Test tensor shape operations
    test_reshape_operation();
    
    println!("All tests completed successfully!");
}

fn test_basic_arithmetic() {
    println!("Testing basic arithmetic operations...");
    
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2, 2]);
    
    // Create input tensors
    let input1 = graph.placeholder(&shape, MPSDataType::Float32, Some("input1"));
    let input2 = graph.placeholder(&shape, MPSDataType::Float32, Some("input2"));
    
    // Create operations
    let add = graph.add(&input1, &input2, Some("add"));
    let sub = graph.subtract(&input1, &input2, Some("sub"));
    let mul = graph.multiply(&input1, &input2, Some("mul"));
    let div = graph.divide(&input1, &input2, Some("div"));
    
    // Create input data
    let data1 = vec![5.0f32, 10.0, 15.0, 20.0];
    let data2 = vec![2.0f32, 4.0, 6.0, 8.0];
    
    let tensor_data1 = MPSGraphTensorData::from_bytes(&data1, &shape, MPSDataType::Float32);
    let tensor_data2 = MPSGraphTensorData::from_bytes(&data2, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&input1, tensor_data1);
    feeds.insert(&input2, tensor_data2);
    
    // Run the graph
    let results = graph.run(feeds, &[&add, &sub, &mul, &div]);
    
    // Verify results
    for (tensor, data) in results {
        match tensor.name().as_str() {
            "add" => {
                let result = data.to_vec::<f32>();
                println!("Addition result: {:?}", result);
                assert_eq!(result, vec![7.0, 14.0, 21.0, 28.0]);
            },
            "sub" => {
                let result = data.to_vec::<f32>();
                println!("Subtraction result: {:?}", result);
                assert_eq!(result, vec![3.0, 6.0, 9.0, 12.0]);
            },
            "mul" => {
                let result = data.to_vec::<f32>();
                println!("Multiplication result: {:?}", result);
                assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);
            },
            "div" => {
                let result = data.to_vec::<f32>();
                println!("Division result: {:?}", result);
                assert_eq!(result, vec![2.5, 2.5, 2.5, 2.5]);
            },
            _ => panic!("Unexpected tensor name: {}", tensor.name()),
        }
    }
    
    println!("Basic arithmetic operations test passed!");
}

fn test_activation_functions() {
    println!("Testing activation functions...");
    
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
    let results = graph.run(feeds, &[&relu, &sigmoid, &tanh]);
    
    // Verify results
    for (tensor, data) in results {
        match tensor.name().as_str() {
            "relu" => {
                let result = data.to_vec::<f32>();
                println!("ReLU result: {:?}", result);
                assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
            },
            "sigmoid" => {
                let result = data.to_vec::<f32>();
                println!("Sigmoid result: {:?}", result);
                // Calculate expected sigmoid values for comparison
                let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
                let expected: Vec<f32> = input_data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                for (a, b) in result.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-6, "Expected {}, got {}", b, a);
                }
            },
            "tanh" => {
                let result = data.to_vec::<f32>();
                println!("Tanh result: {:?}", result);
                // Calculate expected tanh values for comparison
                let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
                let expected: Vec<f32> = input_data.iter().map(|&x| x.tanh()).collect();
                for (a, b) in result.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-6, "Expected {}, got {}", b, a);
                }
            },
            _ => panic!("Unexpected tensor name: {}", tensor.name()),
        }
    }
    
    println!("Activation functions test passed!");
}

fn test_reduction_operations() {
    println!("Testing reduction operations...");
    
    let graph = MPSGraph::new();
    let shape = MPSShape::from_slice(&[2, 3]);
    
    // Create input tensor
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    
    // Create operations
    let sum_cols = graph.reduce_sum(&input, &[1], Some("sum_cols"));
    let mean_cols = graph.reduce_mean(&input, &[1], Some("mean_cols"));
    let max_cols = graph.reduce_max(&input, &[1], Some("max_cols"));
    let min_cols = graph.reduce_min(&input, &[1], Some("min_cols"));
    
    // Create input data - [[1,2,3], [4,5,6]]
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&input, tensor_data);
    
    // Run the graph
    let results = graph.run(feeds, &[&sum_cols, &mean_cols, &max_cols, &min_cols]);
    
    // Verify results
    for (tensor, data) in results {
        match tensor.name().as_str() {
            "sum_cols" => {
                let result = data.to_vec::<f32>();
                println!("Sum reduction result: {:?}", result);
                assert_eq!(result, vec![6.0, 15.0]); // Sum of rows
            },
            "mean_cols" => {
                let result = data.to_vec::<f32>();
                println!("Mean reduction result: {:?}", result);
                assert_eq!(result, vec![2.0, 5.0]); // Mean of rows
            },
            "max_cols" => {
                let result = data.to_vec::<f32>();
                println!("Max reduction result: {:?}", result);
                assert_eq!(result, vec![3.0, 6.0]); // Max of rows
            },
            "min_cols" => {
                let result = data.to_vec::<f32>();
                println!("Min reduction result: {:?}", result);
                assert_eq!(result, vec![1.0, 4.0]); // Min of rows
            },
            _ => panic!("Unexpected tensor name: {}", tensor.name()),
        }
    }
    
    println!("Reduction operations test passed!");
}

fn test_matrix_multiplication() {
    println!("Testing matrix multiplication...");
    
    let graph = MPSGraph::new();
    let shape_a = MPSShape::from_slice(&[2, 3]);
    let shape_b = MPSShape::from_slice(&[3, 2]);
    
    // Create input tensors
    let a = graph.placeholder(&shape_a, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape_b, MPSDataType::Float32, Some("b"));
    
    // Create operation
    let matmul = graph.matmul(&a, &b, Some("matmul"));
    
    // Create input data
    // Matrix A: [[1, 2, 3], [4, 5, 6]]
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: [[1, 2], [3, 4], [5, 6]]
    let data_b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    let tensor_data_a = MPSGraphTensorData::from_bytes(&data_a, &shape_a, MPSDataType::Float32);
    let tensor_data_b = MPSGraphTensorData::from_bytes(&data_b, &shape_b, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&a, tensor_data_a);
    feeds.insert(&b, tensor_data_b);
    
    // Run the graph
    let results = graph.run(feeds, &[&matmul]);
    
    // Verify results
    let (tensor, data) = results.into_iter().next().unwrap();
    assert_eq!(tensor.name(), "matmul");
    
    // Expected result: [[22, 28], [49, 64]]
    let result = data.to_vec::<f32>();
    println!("Matrix multiplication result: {:?}", result);
    assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);
    
    println!("Matrix multiplication test passed!");
}

fn test_reshape_operation() {
    println!("Testing reshape operation...");
    
    let graph = MPSGraph::new();
    let shape_in = MPSShape::from_slice(&[2, 3]);
    let shape_out = MPSShape::from_slice(&[3, 2]);
    
    // Create input tensor
    let input = graph.placeholder(&shape_in, MPSDataType::Float32, Some("input"));
    
    // Create operation
    let reshaped = graph.reshape(&input, &shape_out, Some("reshaped"));
    
    // Create input data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape_in, MPSDataType::Float32);
    
    // Create feed dictionary
    let mut feeds = HashMap::new();
    feeds.insert(&input, tensor_data);
    
    // Run the graph
    let results = graph.run(feeds, &[&reshaped]);
    
    // Verify results
    let (tensor, data) = results.into_iter().next().unwrap();
    assert_eq!(tensor.name(), "reshaped");
    
    // The data values should remain the same, just arranged in a different shape
    let result = data.to_vec::<f32>();
    println!("Reshape result: {:?}", result);
    println!("Result shape: {:?}", data.shape().dimensions());
    
    // Check shape and total elements
    assert_eq!(data.shape().dimensions(), vec![3, 2]);
    assert_eq!(result.len(), 6);
    
    // Sum should remain the same
    assert_eq!(result.iter().sum::<f32>(), 21.0);
    
    println!("Reshape operation test passed!");
}