// Import the minimum required components
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::tensor_data::MPSGraphTensorData;

fn main() {
    println!("Testing MPSGraphTensorData creation");
    
    // Create a simple tensor data object
    let shape = MPSShape::from_slice(&[3, 3]);
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    
    println!("About to create MPSGraphTensorData object...");
    
    // Using the from_bytes method
    println!("  Using from_bytes method...");
    let tensor_data = MPSGraphTensorData::from_bytes(&input_data, &shape, MPSDataType::Float32);
    println!("  Successfully created tensor data with from_bytes!");
    
    // Using the new method
    println!("  Using new method...");
    let tensor_data_new = MPSGraphTensorData::new(&input_data, &[3, 3], MPSDataType::Float32);
    println!("  Successfully created tensor data with new!");
    
    // Test accessing properties
    println!("Data type: {:?}", tensor_data.data_type());
    println!("Shape: {:?}", tensor_data.shape().dimensions());
    
    println!("All tensor data operations completed successfully!");
}