use mpsgraph::{
    MPSGraph,
    MPSDataType,
    MPSShape,
    MPSGraphTensorData,
    MPSGraphExecutionDescriptor,
    MPSCommandBuffer
};
use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;

/// A simple example demonstrating how to use MPSGraph with MPSCommandBuffer
/// 
/// This example shows:
/// 1. Creation of Metal buffers for inputs and outputs
/// 2. Wrapping buffers in MPSGraphTensorData
/// 3. Building a computation graph with 3 operations
/// 4. Creating an MPSCommandBuffer and encoding the graph
/// 5. Committing and waiting for the command buffer
/// 6. Reading results directly from the Metal buffers
fn main() {
    println!("MPSGraph with MPSCommandBuffer example\n");
    
    //-- Setup --//
    
    // Get the default Metal device
    let device = Device::system_default().expect("No Metal device found");
    println!("Using device: {}", device.name());
    
    // Create a command queue for execution
    let command_queue = device.new_command_queue();
    
    // Create a graph to define our computation
    let graph = MPSGraph::new();
    
    //-- Define Computation Graph --//
    println!("Building computation graph...");
    
    // 1. Define input tensors
    let shape = MPSShape::from_slice(&[2, 2]);
    let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&shape, MPSDataType::Float32, Some("B"));
    
    // 2. Define computation operations
    // Operation 1: C = A + B
    let c = graph.add(&a, &b, Some("C"));
    
    // Operation 2: D = C * C (element-wise multiply)
    let d = graph.multiply(&c, &c, Some("D"));
    
    // Operation 3: E = C + D
    let e = graph.add(&c, &d, Some("E"));
    
    //-- Create Buffers for Input and Output Data --//
    println!("Creating Metal buffers for inputs and outputs...");
    
    // Define input data
    let a_input = [1.0f32, 2.0, 3.0, 4.0];
    let b_input = [5.0f32, 6.0, 7.0, 8.0];
    
    // Calculate buffer size for a 2x2 matrix of f32
    let buffer_size = (4 * std::mem::size_of::<f32>()) as u64;
    
    // Create Metal buffers with StorageModeShared for CPU/GPU access
    // Input buffers initialized with data
    let a_buffer = device.new_buffer_with_data(
        a_input.as_ptr() as *const _, 
        buffer_size,
        MTLResourceOptions::StorageModeShared
    );
    
    let b_buffer = device.new_buffer_with_data(
        b_input.as_ptr() as *const _, 
        buffer_size,
        MTLResourceOptions::StorageModeShared
    );
    
    // Output buffers (empty)
    let c_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
    let d_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
    let e_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
    
    //-- Create MPSGraphTensorData from Metal Buffers --//
    println!("Creating MPSGraphTensorData from Metal buffers...");
    
    // Wrap Metal buffers in MPSGraphTensorData
    let a_data = MPSGraphTensorData::from_buffer(&a_buffer, &shape, MPSDataType::Float32);
    let b_data = MPSGraphTensorData::from_buffer(&b_buffer, &shape, MPSDataType::Float32);
    let c_data = MPSGraphTensorData::from_buffer(&c_buffer, &shape, MPSDataType::Float32);
    let d_data = MPSGraphTensorData::from_buffer(&d_buffer, &shape, MPSDataType::Float32);
    let e_data = MPSGraphTensorData::from_buffer(&e_buffer, &shape, MPSDataType::Float32);
    
    //-- Set Up Feeds and Results --//
    println!("Setting up feeds and results dictionaries...");
    
    // Create feed dictionary (inputs)
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);
    
    // Create results dictionary (outputs)
    let mut results = HashMap::new();
    results.insert(c.clone(), c_data);
    results.insert(d.clone(), d_data);
    results.insert(e.clone(), e_data);
    
    //-- Create Execution Descriptor --//
    let execution_descriptor = MPSGraphExecutionDescriptor::new();
    execution_descriptor.prefer_synchronous_execution();
    
    //-- Create MPSCommandBuffer --//
    println!("Creating MPSCommandBuffer...");
    
    // Create an MPSCommandBuffer from the command queue
    let mps_command_buffer = MPSCommandBuffer::from_command_queue(&command_queue);
    
    // Set a label for debugging
    mps_command_buffer.set_label("MPSGraph Simple Compile");
    
    //-- Encode Graph to Command Buffer --//
    println!("Encoding graph to MPSCommandBuffer...");
    
    // Encode the graph operations to the command buffer
    graph.encode_to_command_buffer_with_results(
        &mps_command_buffer,
        &feeds,
        None, // No specific target operations - we'll use the results dictionary instead
        &results,
        Some(&execution_descriptor)
    );
    
    //-- Commit and Wait for Completion --//
    println!("Committing command buffer and waiting for completion...");
    
    // Commit the command buffer
    mps_command_buffer.commit();
    
    // Wait for execution to complete
    mps_command_buffer.wait_until_completed();
    
    // Check command buffer status
    println!("Command buffer status: {:?}", mps_command_buffer.status());
    
    // Check for errors
    if let Some(error) = mps_command_buffer.error() {
        println!("Error during execution: {}", error);
    }
    
    //-- Read Results --//
    println!("\nReading results directly from Metal buffers:");
    
    // Read C = A + B
    let c_result = unsafe {
        let ptr = c_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, 4).to_vec()
    };
    println!("C = A + B:        {:?}", c_result);
    
    // Read D = C * C
    let d_result = unsafe {
        let ptr = d_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, 4).to_vec()
    };
    println!("D = C * C:        {:?}", d_result);
    
    // Read E = C + D
    let e_result = unsafe {
        let ptr = e_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, 4).to_vec()
    };
    println!("E = C + D:        {:?}", e_result);
    
    //-- Verify Results --//
    println!("\nVerifying results:");
    
    // Expected results
    let expected_c = [6.0, 8.0, 10.0, 12.0];          // A + B
    let expected_d = [36.0, 64.0, 100.0, 144.0];      // (A+B) * (A+B)
    let expected_e = [42.0, 72.0, 110.0, 156.0];      // (A+B) + (A+B)*(A+B)
    
    // Check if results match expected values
    let c_correct = c_result.iter().zip(expected_c.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);
    let d_correct = d_result.iter().zip(expected_d.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);
    let e_correct = e_result.iter().zip(expected_e.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);
    
    if c_correct && d_correct && e_correct {
        println!("✅ All results correct!");
    } else {
        println!("❌ Results don't match expected values.");
        if !c_correct { println!("- C expected: {:?}", expected_c); }
        if !d_correct { println!("- D expected: {:?}", expected_d); }
        if !e_correct { println!("- E expected: {:?}", expected_e); }
    }
    
    //-- Performance Info --//
    if mps_command_buffer.gpu_end_time() > 0.0 {
        let gpu_time = mps_command_buffer.gpu_end_time() - mps_command_buffer.gpu_start_time();
        println!("\nGPU execution time: {:.6} ms", gpu_time * 1000.0);
    }
    
    println!("\nExecution complete!");
}