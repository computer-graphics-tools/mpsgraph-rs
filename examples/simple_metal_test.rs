use metal::Device;
use std::collections::HashMap;

// Import the minimum necessary components
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;

fn main() {
    if let Some(device) = Device::system_default() {
        println!("Using Metal device: {}", device.name());
        test_simple_operations();
    } else {
        println!("No Metal device found.");
    }
}

fn test_simple_operations() {
    println!("Testing simple operations on real Metal hardware");
    
    let graph = MPSGraph::new();
    
    // Create a basic 2x2 matrix
    let shape = MPSShape::from_slice(&[2, 2]);
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];
    
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("b"));
    
    // Perform addition
    let add_result = graph.add(&a, &b, Some("add"));
    
    // Run the graph
    let tensor_data_a = MPSGraphTensorData::from_bytes(&data_a, &shape, MPSDataType::Float32);
    let tensor_data_b = MPSGraphTensorData::from_bytes(&data_b, &shape, MPSDataType::Float32);
    
    let mut feeds = HashMap::new();
    feeds.insert(&a, tensor_data_a);
    feeds.insert(&b, tensor_data_b);
    
    let results = graph.run(feeds, &[&add_result]);
    
    // Display and verify the result
    let add_data = results[&add_result].to_vec::<f32>();
    println!("Addition result: {:?}", add_data);
    
    // Expected result: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    assert_eq!(add_data, vec![6.0, 8.0, 10.0, 12.0]);
    
    println!("Test passed - basic addition works on Metal!");
    
    // Now test one of the newer operations: matrix inverse
    test_matrix_inverse();
}

fn test_matrix_inverse() {
    println!("\nTesting matrix inverse operation...");
    
    let graph = MPSGraph::new();
    
    // Create a 2x2 invertible matrix
    let shape = MPSShape::from_slice(&[2, 2]);
    let data = vec![4.0f32, 3.0, 2.0, 1.0]; // Matrix [[4, 3], [2, 1]]
    
    let matrix = graph.placeholder(&shape, MPSDataType::Float32, Some("matrix"));
    
    // Compute the inverse of the matrix
    let inverse = graph.matrix_inverse(&matrix, Some("inverse"));
    
    // Run the graph
    let tensor_data = MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32);
    
    let mut feeds = HashMap::new();
    feeds.insert(&matrix, tensor_data);
    
    let results = graph.run(feeds, &[&inverse]);
    
    // Get the result data
    let inverse_data = results[&inverse].to_vec::<f32>();
    println!("Original matrix: {:?}", data);
    println!("Inverse matrix: {:?}", inverse_data);
    
    // Test that the result is correct
    // For matrix [[4, 3], [2, 1]], the inverse should be [[-0.5, 1.5], [1, -2]]
    assert!((inverse_data[0] + 0.5).abs() < 1e-5);
    assert!((inverse_data[1] - 1.5).abs() < 1e-5);
    assert!((inverse_data[2] - 1.0).abs() < 1e-5);
    assert!((inverse_data[3] + 2.0).abs() < 1e-5);
    
    println!("Test passed - matrix inverse works on Metal!");
}