use mpsgraph::prelude::*;

fn main() {
    // Create a graph
    let graph = MPSGraph::new();
    
    // Create input placeholders
    let a_shape = MPSShape::matrix(2, 3);
    let b_shape = MPSShape::matrix(3, 2);
    
    let a = graph.placeholder(&a_shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&b_shape, MPSDataType::Float32, Some("B"));
    
    // Get graph options
    let options = graph.options();
    
    // Create a compilation descriptor
    let compilation_descriptor = MPSGraphCompilationDescriptor::new();
    compilation_descriptor.set_optimization_level(MPSGraphOptimization::Level1);
    
    // Create an execution descriptor
    let execution_descriptor = MPSGraphExecutionDescriptor::new();
    execution_descriptor.set_wait_until_completed(true);
    execution_descriptor.set_compilation_descriptor(Some(&compilation_descriptor));
    
    println!("Created a mock graph with compilation and execution descriptors");
    println!("Graph options: {:?}", options);
    println!("Input A shape: {:?}", a_shape.dimensions());
    println!("Input B shape: {:?}", b_shape.dimensions());
}