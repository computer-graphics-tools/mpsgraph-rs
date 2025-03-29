/// An example demonstrating how to use MPSCommandBuffer with MPSGraphExecutable
/// 
/// This example shows:
/// 1. Creating an MPSGraph and compiling it to an executable
/// 2. Using MPSCommandBuffer with the executable for encoding
/// 3. Reading the results

use metal::{DeviceRef, MTLResourceOptions};
use mpsgraph_rs::{
    MPSGraph, MPSGraphDevice, MPSCommandBuffer, MPSGraphTensor, MPSGraphTensorData, 
    MPSDataType, MPSGraphExecutableExecutionDescriptor
};
use std::collections::HashMap;

fn main() {
    // Initialize Metal device
    let device = metal::Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let graph_device = MPSGraphDevice::new(&device);
    
    println!("Created Metal device and command queue");
    
    // Create graph
    let graph = MPSGraph::new();
    
    // Define inputs and operations
    let shape = vec![2, 2];
    let a = graph.placeholder_with_shape(&shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder_with_shape(&shape, MPSDataType::Float32, Some("B"));
    let c = graph.multiply(&a, &b, Some("C")); // Use multiply this time
    
    println!("Created graph with multiply operation");
    
    // Compile the graph to an executable
    let mut feeds = HashMap::new();
    let a_shape = mpsgraph_rs::MPSShape::from_slice(&shape);
    let b_shape = mpsgraph_rs::MPSShape::from_slice(&shape);
    
    let a_data = MPSGraphTensorData::new_with_shape(&a_shape, MPSDataType::Float32);
    let b_data = MPSGraphTensorData::new_with_shape(&b_shape, MPSDataType::Float32);
    
    feeds.insert(a.clone(), a_data);
    feeds.insert(b.clone(), b_data);
    
    let executable = graph.compile(&graph_device, &feeds, &[c.clone()], None);
    println!("Compiled graph to executable");
    
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
    
    // Prepare execution
    let inputs = vec![a_tensor_data, b_tensor_data];
    let results = Some(vec![c_tensor_data].as_slice());
    
    // Create execution descriptor
    let execution_descriptor = MPSGraphExecutableExecutionDescriptor::new();
    execution_descriptor.prefer_synchronous_execution();
    
    // Create command buffer from queue
    let mps_command_buffer = MPSCommandBuffer::from_command_queue(&command_queue);
    println!("Created MPSCommandBuffer from command queue");
    
    // Encode executable to command buffer
    let _output_data = executable.encode_to_command_buffer(
        &mps_command_buffer,
        &inputs,
        results,
        Some(&execution_descriptor)
    );
    
    println!("Encoded executable to command buffer");
    
    // Get the Metal command buffer and commit it
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
    println!("C (A * B): {:?}", output_slice);
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