use std::collections::HashMap;

// Import just the basic components
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;

fn main() {
    println!("Testing basic operations");
    
    let graph = MPSGraph::new();
    
    // Basic arithmetic example
    test_addition(&graph);
}

fn test_addition(graph: &MPSGraph) {
    println!("Testing basic addition operation...");
    
    // Create input tensors
    let shape = MPSShape::from_slice(&[2, 2]);
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];
    
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("a"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("b"));
    
    // Perform addition
    let add_result = graph.add(&a, &b, Some("add"));
    
    // Prepare input data
    let tensor_data_a = MPSGraphTensorData::from_bytes(&data_a, &shape, MPSDataType::Float32);
    let tensor_data_b = MPSGraphTensorData::from_bytes(&data_b, &shape, MPSDataType::Float32);
    
    // Feed the input data
    let mut feeds = HashMap::new();
    feeds.insert(&a, tensor_data_a);
    feeds.insert(&b, tensor_data_b);
    
    // Run the graph
    let results = graph.run(feeds, &[&add_result]);
    
    // Check the result
    let add_data = results[&add_result].to_vec::<f32>();
    println!("Addition result: {:?}", add_data);
    
    // Expected result: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    assert_eq!(add_data, vec![6.0, 8.0, 10.0, 12.0]);
    
    println!("Test passed!");
}