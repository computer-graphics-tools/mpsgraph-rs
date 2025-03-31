use crate::{
    core::MPSDataType,
    graph::MPSGraph,
    shape::MPSShape,
    tensor_data::MPSGraphTensorData,
};
use std::collections::HashMap;

#[test]
fn test_graph_creation_destruction() {
    // Create a new graph
    let graph = MPSGraph::new();
    
    // Make sure it doesn't crash when dropped
    drop(graph);
    
    // Create and drop multiple graphs
    for _ in 0..5 {
        let _graph = MPSGraph::new();
    }
}

#[test]
fn test_graph_placeholder() {
    let graph = MPSGraph::new();
    
    // Test creating placeholders with various shapes and data types
    let shape1 = MPSShape::from_slice(&[2, 3]);
    let placeholder1 = graph.placeholder(&shape1, MPSDataType::Float32, Some("placeholder1"));
    
    let shape2 = MPSShape::from_slice(&[5, 5, 3]);
    let placeholder2 = graph.placeholder(&shape2, MPSDataType::Int32, Some("placeholder2"));
    
    // Test with no name provided
    let placeholder3 = graph.placeholder(&shape1, MPSDataType::Float16, None);
    
    // Verify properties
    assert_eq!(placeholder1.data_type(), MPSDataType::Float32);
    assert_eq!(placeholder2.data_type(), MPSDataType::Int32);
    assert_eq!(placeholder3.data_type(), MPSDataType::Float16);
    
    // Check shape dimensions
    let p1_shape = placeholder1.shape();
    assert_eq!(p1_shape.dimensions(), vec![2, 3]);
    
    let p2_shape = placeholder2.shape();
    assert_eq!(p2_shape.dimensions(), vec![5, 5, 3]);
}

#[test]
fn test_graph_constant() {
    let graph = MPSGraph::new();
    
    // Create constants with different data types and shapes
    let shape1 = MPSShape::from_slice(&[2, 2]);
    
    // Float32 constant
    let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
    let constant_f32 = graph.constant(&data_f32, &shape1, MPSDataType::Float32);
    
    // Int32 constant
    let data_i32 = vec![5i32, 6, 7, 8];
    let constant_i32 = graph.constant(&data_i32, &shape1, MPSDataType::Int32);
    
    // Verify properties
    assert_eq!(constant_f32.data_type(), MPSDataType::Float32);
    assert_eq!(constant_i32.data_type(), MPSDataType::Int32);
    
    // Check shape
    let shape_f32 = constant_f32.shape();
    assert_eq!(shape_f32.dimensions(), vec![2, 2]);
}

#[test]
fn test_graph_run_with_feeds() {
    let graph = MPSGraph::new();
    
    // Create input placeholders
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("B"));
    
    // Define operations: C = A + B
    let result = graph.add(&a, &b, None);
    
    // Create input data
    let a_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], MPSDataType::Float32);
    let b_data = MPSGraphTensorData::new(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[result.clone()]);
    
    // Verify results
    assert!(results.contains_key(&result), "Results should contain the output tensor");
    
    // Verify results exist
    assert!(results.contains_key(&result), "Results should contain the output tensor");
}

#[test]
fn test_device_specific_run() {
    // Skip if Metal device not available
    if let Some(_metal_device) = metal::Device::system_default() {
        let graph = MPSGraph::new();
        
        // Create input placeholder
        let shape = MPSShape::from_slice(&[2, 2]);
        let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
        
        // Define operation: B = A * 2
        let scalar = graph.constant_scalar(2.0f32, MPSDataType::Float32);
        let result = graph.multiply(&a, &scalar, None);
        
        // Create input data
        let a_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], MPSDataType::Float32);
        
        // Create feeds
        let mut feeds = HashMap::new();
        feeds.insert(a.clone(), a_data);
        
        // Run graph without specifying a device - onDevice method is not available
        let results = graph.run_with_feeds(&feeds, &[result.clone()]);
        
        // Verify results
        assert!(results.contains_key(&result), "Results should contain the output tensor");
        
        // Just verify we got a result
        assert!(results.contains_key(&result));
    }
}

#[test]
fn test_graph_clone() {
    let graph = MPSGraph::new();
    
    // Test that graphs can be cloned without crashing
    let graph_clone = graph.clone();
    
    // Use both graphs
    let shape = MPSShape::from_slice(&[2, 2]);
    let _a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
    let _b = graph_clone.placeholder(&shape, MPSDataType::Float32, Some("B"));
    
    // Both should be dropped without issues
}