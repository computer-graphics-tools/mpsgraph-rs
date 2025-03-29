use mpsgraph::{
    core::MPSDataType,
    executable::MPSGraphExecutable,
    executable::MPSGraphCompilationDescriptor,
    tensor_data::MPSGraphTensorData,
    shape::MPSShape,
};
use metal::{Device, CommandQueue};
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
fn run_coreml_model(model_path: &str, _device: &Device, _command_queue: &CommandQueue) {
    println!("Loading CoreML model from: {}", model_path);
    
    // Create a file URL from the path
    let file_url = format!("file://{}", model_path);
    
    // Create compilation descriptor
    let compilation_descriptor = MPSGraphCompilationDescriptor::new();
    
    // Load executable from CoreML model
    // Note: This will only work on iOS 18/macOS 15+
    let executable = MPSGraphExecutable::from_coreml_package(&file_url, Some(&compilation_descriptor));
    
    if let Some(_exec) = executable {
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
    let _shape = MPSShape::from_slice(&[5, 5]);
    let tensor_data = MPSGraphTensorData::new(&data, &[5, 5], MPSDataType::Float32);
    
    println!("Created tensor with shape {:?} and data type {:?}", 
             tensor_data.shape().dimensions(), tensor_data.data_type());
    
    // The synchronization methods are demonstration-only and might not be 
    // available in all versions of the Metal API
    println!("Note: Synchronization methods simulated for demonstration");
    println!("In a real implementation, these would ensure GPU data is accessible from CPU");
    
    // Instead of synchronizing, we'll just wait a moment
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Access the underlying data directly
    let buffer = unsafe {
        let ptr = tensor_data.0;
        let data_ptr: *mut std::ffi::c_void = objc2::msg_send![ptr, mpsndArrayData];
        if !data_ptr.is_null() {
            println!("Successfully accessed tensor data buffer");
        } else {
            println!("Failed to access tensor data buffer");
        }
    };
    println!("(Simulated) Synchronized tensor to specified device");
    
    println!("Demonstration complete");
}