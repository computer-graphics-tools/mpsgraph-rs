use mpsgraph::prelude::*;
use mpsgraph::executable::MPSGraphExecutionDescriptor;
use metal::{Device, MTLResourceOptions, Buffer};
use std::collections::HashMap;

// A struct that pairs an MTLBuffer with its MPSGraphTensorData
#[derive(Clone)]
struct TensorBuffer {
    buffer: Buffer,
    tensor_data: MPSGraphTensorData,
}

impl TensorBuffer {
    // Create a new TensorBuffer from a vector of f32 data
    fn new(device: &Device, data: &[f32], shape: &MPSShape, data_type: MPSDataType) -> Self {
        // Calculate size in bytes
        let byte_length = data.len() * std::mem::size_of::<f32>();
        
        // Create MTLBuffer with storage mode shared for CPU/GPU access
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        // Create tensor data that references this buffer
        let tensor_data = MPSGraphTensorData::from_buffer(&buffer, shape, data_type);
        
        Self { buffer, tensor_data }
    }
    
    // Create an empty TensorBuffer for results
    fn new_empty(device: &Device, size: usize, shape: &MPSShape, data_type: MPSDataType) -> Self {
        // Calculate size in bytes
        let byte_length = size * std::mem::size_of::<f32>();
        
        // Create empty MTLBuffer with storage mode shared
        let buffer = device.new_buffer(
            byte_length as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        // Create tensor data that references this buffer
        let tensor_data = MPSGraphTensorData::from_buffer(&buffer, shape, data_type);
        
        Self { buffer, tensor_data }
    }
    
    // Read data directly from the buffer
    fn get_f32_data(&self, count: usize) -> Vec<f32> {
        let ptr = self.buffer.contents() as *const f32;
        unsafe {
            std::slice::from_raw_parts(ptr, count).to_vec()
        }
    }
}

fn main() {
    // Get Metal device
    let device = Device::system_default().expect("No Metal device found");
    
    // Input dimensions
    let m = 2;  // Matrix A rows
    let k = 3;  // Matrix A cols / Matrix B rows
    let n = 2;  // Matrix B cols
    
    // Create input data
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 2x3 matrix
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 3x2 matrix
    
    // Create a graph
    let graph = MPSGraph::new();
    
    // Create input placeholders
    let a_shape = MPSShape::matrix(m, k);
    let b_shape = MPSShape::matrix(k, n);
    let result_shape = MPSShape::matrix(m, n);
    
    let a = graph.placeholder(&a_shape, MPSDataType::Float32, Some("A"));
    let b = graph.placeholder(&b_shape, MPSDataType::Float32, Some("B"));
    
    // Perform matrix multiplication: A * B
    let result = graph.matmul(&a, &b, None);
    
    // Create TensorBuffers for inputs
    let a_tensor = TensorBuffer::new(&device, &a_data, &a_shape, MPSDataType::Float32);
    let b_tensor = TensorBuffer::new(&device, &b_data, &b_shape, MPSDataType::Float32);
    
    // Create TensorBuffer for output
    let result_size = m * n;
    let result_tensor = TensorBuffer::new_empty(
        &device, 
        result_size,
        &result_shape, 
        MPSDataType::Float32
    );
    
    // Create command queue
    let command_queue = device.new_command_queue();
    
    // Prepare feeds and targets
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_tensor.tensor_data);
    feeds.insert(&b, &b_tensor.tensor_data);
    
    println!("Executing matrix multiplication...");
    
    let mut output_map = HashMap::new();
    output_map.insert(&result, &result_tensor.tensor_data);
    
    // Execute graph with our inputs and output buffers
    // Create execution descriptor that waits until completed
    let execution_descriptor = MPSGraphExecutionDescriptor::new();
    execution_descriptor.set_wait_until_completed(true);
    
    graph.run_with_command_queue_feeds_outputs(
        &command_queue,
        feeds,
        output_map,
        Some(&execution_descriptor)
    );
    
    // Now read the result directly from the MTLBuffer
    println!("Reading result from buffer...");
    let result_floats = result_tensor.get_f32_data(result_size);
    println!("Result data: {:?}", result_floats);
    
    // Display the matrices and result
    println!("Matrix A ({m}x{k}):");
    for i in 0..m {
        let row: Vec<f32> = a_data[i*k..(i+1)*k].to_vec();
        println!("  {:?}", row);
    }
    
    println!("Matrix B ({k}x{n}):");
    for i in 0..k {
        let row: Vec<f32> = b_data[i*n..(i+1)*n].to_vec();
        println!("  {:?}", row);
    }
    
    println!("Result ({m}x{n}):");
    for i in 0..m {
        let row: Vec<f32> = result_floats[i*n..(i+1)*n].to_vec();
        println!("  {:?}", row);
    }
}