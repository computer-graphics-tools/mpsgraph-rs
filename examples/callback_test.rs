use mpsgraph::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::thread;

fn main() {
    // Create the graph and device
    let graph = MPSGraph::new();
    
    // Create input placeholders
    let a = graph.placeholder(&[2, 2], MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&[2, 2], MPSDataType::Float32, Some("B"));
    
    // Define operations: C = A + B
    let result = graph.add(&a, &b, None);
    
    // Create some input data
    let a_data = MPSGraphTensorData::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], MPSDataType::Float32);
    let b_data = MPSGraphTensorData::new(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], MPSDataType::Float32);
    
    // Create input feeds
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);
    
    // Get a Metal command queue
    let metal_device = metal::Device::system_default().unwrap();
    let command_queue = metal_device.new_command_queue();
    
    // Create execution descriptor with asynchronous execution preference
    let execution_descriptor = MPSGraphExecutionDescriptor::new();
    execution_descriptor.prefer_asynchronous_execution();
    
    // Create a flag to signal when the callback has completed
    let callback_completed = Arc::new(Mutex::new(false));
    let callback_completed_clone = Arc::clone(&callback_completed);
    
    println!("Starting asynchronous execution (simulated with a callback)...");
    
    // Run with simulated async callback
    let async_result = graph.run_async_with_command_queue(
        &command_queue,
        &feeds,
        &[result.clone()],
        &execution_descriptor,
        Some(move |results: HashMap<MPSGraphTensor, MPSGraphTensorData>| {
            println!("Callback received results!");
            
            // Get the result tensor data
            let result_data = results.get(&result).unwrap();
            
            // Access the result data (depends on how you want to read the data)
            // For demonstration, we'll just show we have the result
            println!("Result tensor data received with shape: {:?}", result_data.shape().dimensions());
            
            // Signal that callback has completed
            *callback_completed_clone.lock().unwrap() = true;
        })
    );
    
    // Wait for the callback to be completed
    let mut attempts = 0;
    while !*callback_completed.lock().unwrap() && attempts < 10 {
        println!("Waiting for callback to complete...");
        thread::sleep(Duration::from_millis(100));
        attempts += 1;
    }
    
    println!("Immediate result is also available: {:?}", async_result.contains_key(&result));
    
    // For comparison, run synchronously
    let sync_descriptor = MPSGraphExecutionDescriptor::new();
    sync_descriptor.prefer_synchronous_execution();
    
    println!("\nRunning synchronously for comparison:");
    let sync_result = graph.run_with_feeds_and_descriptor(
        &feeds,
        &[result.clone()], 
        &sync_descriptor
    );
    
    println!("Synchronous result is available: {:?}", sync_result.contains_key(&result));
    
    println!("Callback test completed!");
}