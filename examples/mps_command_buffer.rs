/// An example demonstrating how to use MPSCommandBuffer with MPSGraph
/// 
/// This example shows:
/// 1. Creating an MPSCommandBuffer from a command queue
/// 2. Using MPSCommandBuffer with MPSGraph encoding
/// 3. Proper command buffer management

use metal::{DeviceRef, CommandQueue, MTLResourceOptions};
use mpsgraph_rs::{
    MPSGraph, MPSCommandBuffer, MPSGraphTensor, MPSGraphTensorData, MPSDataType, MPSGraphExecutionDescriptor
};
use std::collections::HashMap;

fn main() {
    // Initialize Metal device
    let device = metal::Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    
    // Create graph
    let graph = MPSGraph::new();
    
    // Define inputs and operations
    let shape = vec![2, 2];
    let a = graph.placeholder_with_shape(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder_with_shape(&shape, MPSDataType::Float32, Some("B"));
    let c = graph.add(&a, &b, Some("C"));
    
    println!("Created graph with add operation");
    
    // Create input data
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];
    
    let a_tensor_data = MPSGraphTensorData::from_buffer_with_shape(
        &create_buffer_from_data(&device, &a_data),
        &shape,
        MPSDataType::Float32
    );
    
    let b_tensor_data = MPSGraphTensorData::from_buffer_with_shape(
        &create_buffer_from_data(&device, &b_data),
        &shape,
        MPSDataType::Float32
    );
    
    // Create output buffer and tensor data
    let output_buffer_size = (4 * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.new_buffer(output_buffer_size, MTLResourceOptions::StorageModeShared);
    
    let c_tensor_data = MPSGraphTensorData::from_buffer(
        &output_buffer,
        &shape,
        MPSDataType::Float32
    );
    
    println!("Created input and output tensor data");
    
    // Set up feeds and results
    let mut feeds = HashMap::new();
    feeds.insert(a.clone(), a_tensor_data);
    feeds.insert(b.clone(), b_tensor_data);
    
    let mut results = HashMap::new();
    results.insert(c.clone(), c_tensor_data);
    
    // Create execution descriptor
    let execution_descriptor = MPSGraphExecutionDescriptor::new();
    execution_descriptor.prefer_synchronous_execution();
    
    println!("Configured execution descriptor");
    
    // Create command buffer - this is the MPSCommandBuffer approach
    // Option 1: From a Metal CommandBuffer
    let metal_command_buffer = command_queue.new_command_buffer();
    let mps_command_buffer = MPSCommandBuffer::from_command_buffer(&metal_command_buffer);
    
    // Option 2: Directly from CommandQueue
    // let mps_command_buffer = MPSCommandBuffer::from_command_queue(&command_queue);
    
    println!("Created MPSCommandBuffer");
    
    // Encode operations to command buffer
    graph.encode_to_command_buffer_with_results(
        &mps_command_buffer,
        &feeds,
        Some(&[&c]),
        &results,
        Some(&execution_descriptor)
    );
    
    println!("Encoded graph to command buffer");
    
    // Commit the Metal command buffer
    let metal_cmd_buffer = mps_command_buffer.command_buffer();
    metal_cmd_buffer.commit();
    metal_cmd_buffer.wait_until_completed();
    
    println!("Committed and waited for command buffer");
    
    // Read results
    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe {
        std::slice::from_raw_parts(output_ptr, 4)
    };
    
    println!("Results:");
    println!("A: {:?}", a_data);
    println!("B: {:?}", b_data);
    println!("C (A + B): {:?}", output_slice);
}

// Helper function to create a Metal buffer from data
fn create_buffer_from_data(device: &DeviceRef, data: &[f32]) -> metal::Buffer {
    let buffer_size = (data.len() * std::mem::size_of::<f32>()) as u64;
    
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared
    )
}