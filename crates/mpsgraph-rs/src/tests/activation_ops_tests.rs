use crate::{
    core::MPSDataType,
    graph::MPSGraph,
    shape::MPSShape,
    tensor_data::MPSGraphTensorData,
    tensor::MPSGraphTensor,
};
use std::collections::HashMap;

// Simple verification helper that checks if result exists
fn verify_result_exists(results: &HashMap<MPSGraphTensor, MPSGraphTensorData>, tensor: &MPSGraphTensor, name: &str) {
    assert!(results.contains_key(tensor), "Results should contain {} tensor", name);
}

#[test]
fn test_relu_activation() {
    // Use an empty scope to force destructors to run earlier
    {
        let graph = MPSGraph::new();
        println!("DEBUG: Created MPSGraph");
        
        // Create input
        let shape = MPSShape::from_slice(&[2, 3]);
        println!("DEBUG: Created MPSShape");
        
        let input = graph.placeholder(&shape, MPSDataType::Float32, None); // No name for now
        println!("DEBUG: Created placeholder");
        
        // Define ReLU operation without a name to reduce complexity
        let relu_result = graph.relu(&input, None);
        println!("DEBUG: Created ReLU operation");
        
        // Create input data with both positive and negative values
        let input_data = MPSGraphTensorData::new(
            &[-1.0f32, 0.0, 1.0, -2.0, 0.5, 3.0],
            &[2, 3],
            MPSDataType::Float32,
        );
        println!("DEBUG: Created input data");
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(input.clone(), input_data);
        println!("DEBUG: Created feeds");
        
        // Run graph
        let results = graph.run_with_feeds(&feeds, &[relu_result.clone()]);
        println!("DEBUG: Ran graph");
        
        // Verify ReLU results
        verify_result_exists(&results, &relu_result, "ReLU");
        println!("DEBUG: Verified results");
        
        // Force drop of local variables
        std::mem::drop(results);
        std::mem::drop(feeds);
        std::mem::drop(relu_result);
        std::mem::drop(input);
        std::mem::drop(shape);
        std::mem::drop(graph);
        println!("DEBUG: Explicitly dropped all variables");
    }
    
    println!("DEBUG: Outer scope finished");
}

#[test]
fn test_sigmoid_activation() {
    // Use an empty scope to force destructors to run earlier
    {
        let graph = MPSGraph::new();
        println!("DEBUG: Created MPSGraph");
        
        // Create input
        let shape = MPSShape::from_slice(&[2, 2]);
        println!("DEBUG: Created MPSShape");
        
        let input = graph.placeholder(&shape, MPSDataType::Float32, None); // No name
        println!("DEBUG: Created placeholder");
        
        // Define sigmoid operation - no name
        let sigmoid_result = graph.sigmoid(&input, None);
        println!("DEBUG: Created sigmoid operation");
        
        // Create input data
        let input_data = MPSGraphTensorData::new(&[-2.0f32, -1.0, 0.0, 2.0], &[2, 2], MPSDataType::Float32);
        println!("DEBUG: Created input data");
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(input.clone(), input_data);
        println!("DEBUG: Created feeds");
        
        // Run graph
        let results = graph.run_with_feeds(&feeds, &[sigmoid_result.clone()]);
        println!("DEBUG: Ran graph");
        
        // Verify sigmoid results
        verify_result_exists(&results, &sigmoid_result, "sigmoid");
        println!("DEBUG: Verified results");
        
        // Force drop of local variables
        std::mem::drop(results);
        std::mem::drop(feeds);
        std::mem::drop(sigmoid_result);
        std::mem::drop(input);
        std::mem::drop(shape);
        std::mem::drop(graph);
        println!("DEBUG: Explicitly dropped all variables");
    }
    
    println!("DEBUG: Outer scope finished");
}

#[test]
fn test_tanh_activation() {
    // Use an empty scope to force destructors to run earlier
    {
        let graph = MPSGraph::new();
        println!("DEBUG: Created MPSGraph");
        
        // Create input
        let shape = MPSShape::from_slice(&[2, 2]);
        println!("DEBUG: Created MPSShape");
        
        let input = graph.placeholder(&shape, MPSDataType::Float32, None); // No name
        println!("DEBUG: Created placeholder");
        
        // Define tanh operation - no name
        let tanh_result = graph.tanh(&input, None);
        println!("DEBUG: Created tanh operation");
        
        // Create input data
        let input_data = MPSGraphTensorData::new(&[-2.0f32, -1.0, 0.0, 2.0], &[2, 2], MPSDataType::Float32);
        println!("DEBUG: Created input data");
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(input.clone(), input_data);
        println!("DEBUG: Created feeds");
        
        // Run graph
        let results = graph.run_with_feeds(&feeds, &[tanh_result.clone()]);
        println!("DEBUG: Ran graph");
        
        // Verify tanh results
        verify_result_exists(&results, &tanh_result, "tanh");
        println!("DEBUG: Verified results");
        
        // Force drop of local variables
        std::mem::drop(results);
        std::mem::drop(feeds);
        std::mem::drop(tanh_result);
        std::mem::drop(input);
        std::mem::drop(shape);
        std::mem::drop(graph);
        println!("DEBUG: Explicitly dropped all variables");
    }
    
    println!("DEBUG: Outer scope finished");
}

#[test]
fn test_softmax_activation() {
    // Use an empty scope to force destructors to run earlier
    {
        let graph = MPSGraph::new();
        println!("DEBUG: Created MPSGraph");
        
        // Create input
        let shape = MPSShape::from_slice(&[2, 3]);
        println!("DEBUG: Created MPSShape");
        
        let input = graph.placeholder(&shape, MPSDataType::Float32, None); // No name
        println!("DEBUG: Created placeholder");
        
        // Define softmax operation with axis=1 (softmax across each row) - no name
        let softmax_result = graph.softmax(&input, 1, None);
        println!("DEBUG: Created softmax operation");
        
        // Create input data
        let input_data = MPSGraphTensorData::new(
            &[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0],
            &[2, 3],
            MPSDataType::Float32,
        );
        println!("DEBUG: Created input data");
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(input.clone(), input_data);
        println!("DEBUG: Created feeds");
        
        // Run graph
        let results = graph.run_with_feeds(&feeds, &[softmax_result.clone()]);
        println!("DEBUG: Ran graph");
        
        // Verify softmax results
        verify_result_exists(&results, &softmax_result, "softmax");
        println!("DEBUG: Verified results");
        
        // Force drop of local variables
        std::mem::drop(results);
        std::mem::drop(feeds);
        std::mem::drop(softmax_result);
        std::mem::drop(input);
        std::mem::drop(shape);
        std::mem::drop(graph);
        println!("DEBUG: Explicitly dropped all variables");
    }
    
    println!("DEBUG: Outer scope finished");
}

#[test]
fn test_leaky_relu_activation() {
    // Use an empty scope to force destructors to run earlier
    {
        let graph = MPSGraph::new();
        println!("DEBUG: Created MPSGraph");
        
        // Create input
        let shape = MPSShape::from_slice(&[2, 3]);
        println!("DEBUG: Created MPSShape");
        
        let input = graph.placeholder(&shape, MPSDataType::Float32, None); // No name
        println!("DEBUG: Created placeholder");
        
        // Define leaky ReLU operation with alpha=0.1 - no name
        let alpha = 0.1;
        let leaky_relu_result = graph.leaky_relu(&input, alpha, None);
        println!("DEBUG: Created leaky ReLU operation");
        
        // Create input data with both positive and negative values
        let input_data = MPSGraphTensorData::new(
            &[-1.0f32, 0.0, 1.0, -2.0, 0.5, 3.0],
            &[2, 3],
            MPSDataType::Float32,
        );
        println!("DEBUG: Created input data");
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(input.clone(), input_data);
        println!("DEBUG: Created feeds");
        
        // Run graph
        let results = graph.run_with_feeds(&feeds, &[leaky_relu_result.clone()]);
        println!("DEBUG: Ran graph");
        
        // Verify leaky ReLU results
        verify_result_exists(&results, &leaky_relu_result, "leaky ReLU");
        println!("DEBUG: Verified results");
        
        // Force drop of local variables
        std::mem::drop(results);
        std::mem::drop(feeds);
        std::mem::drop(leaky_relu_result);
        std::mem::drop(input);
        std::mem::drop(shape);
        std::mem::drop(graph);
        println!("DEBUG: Explicitly dropped all variables");
    }
    
    println!("DEBUG: Outer scope finished");
}