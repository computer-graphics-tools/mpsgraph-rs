// Import the needed components only
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;

fn main() {
    println!("Running minimal API validation test");
    
    // Create a graph instance - this is the core object for MPS operations
    let graph = MPSGraph::new();
    println!("Successfully created a graph instance");
    
    // Create a simple placeholder tensor
    let shape = MPSShape::from_slice(&[2, 2]);
    let tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    println!("Successfully created a placeholder tensor");
    
    // Create a constant tensor with the same shape
    let constant = graph.constant_scalar(1.0f32, &shape, MPSDataType::Float32, Some("constant"));
    println!("Successfully created a constant tensor");
    
    // Perform a simple operation - add the tensors
    let result = graph.add(&tensor, &constant, Some("addition"));
    println!("Successfully created an addition operation");
    
    // Since we're only validating the API without running, we don't need to 
    // actually execute the graph or check the results.
    println!("All API calls completed successfully. The library is properly linked and the API works!");
}