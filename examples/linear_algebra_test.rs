use std::collections::HashMap;

// Import core components 
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;

fn main() {
    println!("Testing linear algebra operations (matrix inverse)");
    
    // Create graph and input tensors
    let graph = MPSGraph::new();
    
    // Create a 3x3 invertible matrix
    let shape = MPSShape::from_slice(&[3, 3]);
    let input_data = vec![
        1.0f32, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]; // Identity matrix
    
    // Create input tensor
    let input_tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    
    // Create matrix inverse operation
    let inverse_tensor = graph.matrix_inverse(&input_tensor, Some("inverse"));
    
    println!("Successfully created matrix_inverse operation");
    
    // Create input tensor data
    let input_tensor_data = MPSGraphTensorData::from_bytes(&input_data, &shape, MPSDataType::Float32);
    
    println!("Successfully created tensor data");
    
    // Set up feed dictionary
    let mut feed_dict = HashMap::new();
    feed_dict.insert(&input_tensor, input_tensor_data);
    
    println!("Successfully created feed dictionary");
    
    // Run the graph
    let results = graph.run(feed_dict, &[&inverse_tensor]);
    
    println!("Successfully ran the graph");
    
    // Get results
    let inverse_data = results[&inverse_tensor].to_vec::<f32>();
    
    println!("Result shape: {:?}", results[&inverse_tensor].shape().dimensions());
    println!("Inverse result: {:?}", inverse_data);
    
    // For the identity matrix, the inverse should still be the identity matrix
    assert_eq!(inverse_data, input_data);
    
    println!("Matrix inverse test passed!");
}