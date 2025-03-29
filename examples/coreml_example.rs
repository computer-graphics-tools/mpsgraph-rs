use mpsgraph_rs::{
    core::MPSDataType,
    executable::MPSGraphExecutable,
    executable::MPSGraphCompilationDescriptor,
    tensor_data::MPSGraphTensorData,
    tensor::MPSGraphTensor,
    shape::MPSShape,
};
use metal::{Device, CommandQueue};
use std::collections::HashMap;
use std::convert::AsRef;
use std::path::Path;

fn main() {
    // Get the default Metal device
    let device = Device::system_default().expect("No Metal device found");
    println!("Using device: {}", device.name());
    
    // Create a command queue
    let command_queue = device.new_command_queue();
    
    // Example path to a CoreML model package
    // Note: This is just a placeholder - replace with a real path to a .mlmodel or .mlmodelc
    let model_path = "/path/to/your/model.mlmodelc";
    
    // Only run the CoreML example if the model exists
    if Path::new(model_path).exists() {
        run_coreml_model(model_path, &device, &command_queue);
    } else {
        println!("CoreML model not found at {}. Skipping example.", model_path);
        println!("To use this example, replace the model_path with a valid CoreML model.");
        
        // Instead, show a demo of the advanced synchronization features
        demo_advanced_synchronization(&device);
    }
}

/// Run a CoreML model using MPSGraphExecutable
fn run_coreml_model(model_path: &str, device: &Device, command_queue: &CommandQueue) {
    println!("Loading CoreML model from: {}", model_path);
    
    // Create a file URL from the path
    let file_url = format!("file://{}", model_path);
    
    // Create compilation descriptor
    let compilation_descriptor = MPSGraphCompilationDescriptor::new();
    
    // Load executable from CoreML model
    // Note: This will only work on iOS 18/macOS 15+
    let executable = MPSGraphExecutable::from_coreml_package(&file_url, Some(&compilation_descriptor));
    
    if let Some(exec) = executable {
        println!("Successfully loaded CoreML model!");
        
        // Normally, you would:
        // 1. Create input tensors based on model requirements
        // 2. Run the model with those inputs
        // 3. Process the output
        
        println!("CoreML model loaded but example execution skipped (no input data provided)");
    } else {
        println!("Failed to load CoreML model. This could be because:");
        println!("- The model doesn't exist at the specified path");
        println!("- You're running on an OS version older than iOS 18/macOS 15");
        println!("- The model format is incompatible");
    }
}

/// Demonstrate advanced synchronization features
fn demo_advanced_synchronization(device: &Device) {
    println!("\nDemonstrating advanced synchronization features:");
    
    // Create a simple 2D tensor with values from 0 to 24
    let data: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let shape = MPSShape::from_slice(&[5, 5]);
    let tensor_data = MPSGraphTensorData::new(&data, &[5, 5], MPSDataType::Float32);
    
    println!("Created tensor with shape {:?} and data type {:?}", 
             tensor_data.shape().dimensions(), tensor_data.data_type());
    
    // Synchronize entire tensor (default behavior)
    tensor_data.synchronize();
    println!("Synchronized entire tensor to CPU");
    
    // Synchronize just a specific region (2x2 from position 1,1)
    let success = tensor_data.synchronize_region(1, 2, Some(1), Some(2), None, None);
    if success {
        println!("Synchronized 2x2 region starting at (1,1)");
    } else {
        println!("Failed to synchronize region");
    }
    
    // Access the synchronized data
    if let Some(data_slice) = tensor_data.synchronized_data::<f32>() {
        println!("Accessed tensor data as slice, first few values: {:?}", 
                 &data_slice[0..std::cmp::min(5, data_slice.len())]);
    } else {
        println!("Failed to access synchronized data");
    }
    
    // Synchronize to a specific device
    tensor_data.synchronize_to_device(device);
    println!("Synchronized tensor to specified device");
    
    println!("Demonstration complete");
}