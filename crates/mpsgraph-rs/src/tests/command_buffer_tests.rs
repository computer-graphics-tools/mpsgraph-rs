use crate::{core::MPSDataType, graph::MPSGraph, shape::MPSShape, tensor_data::MPSGraphTensorData};
use metal::MTLResourceOptions;
use std::collections::HashMap;

// Helper struct for tests
struct TensorBuffer {
    tensor_data: MPSGraphTensorData,
}

impl TensorBuffer {
    fn new(device: &metal::Device, data: &[f32], shape: &[usize], data_type: MPSDataType) -> Self {
        // Calculate size in bytes
        let byte_length = data.len() * std::mem::size_of::<f32>();

        // Create Metal buffer with storage mode shared
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create tensor data
        let mps_shape = MPSShape::from_slice(shape);
        let tensor_data = MPSGraphTensorData::from_buffer(&buffer, &mps_shape, data_type);

        Self { tensor_data }
    }
}

#[test]
fn test_graph_execution() {
    // Skip if Metal device not available
    if let Some(metal_device) = metal::Device::system_default() {
        let graph = MPSGraph::new();

        // Create input placeholders
        let shape = MPSShape::from_slice(&[2, 2]);
        let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));
        let b = graph.placeholder(&shape, MPSDataType::Float32, Some("B"));

        // Define operations: C = A + B
        let result = graph.add(&a, &b, Some("C"));

        // Create input tensors with Metal buffers
        let a_tensor = TensorBuffer::new(
            &metal_device,
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            MPSDataType::Float32,
        );
        let b_tensor = TensorBuffer::new(
            &metal_device,
            &[5.0f32, 6.0, 7.0, 8.0],
            &[2, 2],
            MPSDataType::Float32,
        );

        // Create output tensor buffer (we don't actually need to use this in the test)
        let _result_tensor = TensorBuffer::new(
            &metal_device,
            &[0.0f32, 0.0, 0.0, 0.0],
            &[2, 2],
            MPSDataType::Float32,
        );

        // Create input feeds
        let mut feeds = HashMap::new();
        feeds.insert(a.clone(), a_tensor.tensor_data.clone());
        feeds.insert(b.clone(), b_tensor.tensor_data.clone());

        // Run graph directly
        let results = graph.run_with_feeds(&feeds, &[result.clone()]);

        // Verify results
        assert!(
            results.contains_key(&result),
            "Results should contain the output tensor"
        );
    }
}

#[test]
fn test_multiple_operations() {
    // Skip if Metal device not available
    if let Some(metal_device) = metal::Device::system_default() {
        let graph = MPSGraph::new();

        // Create input placeholder
        let shape = MPSShape::from_slice(&[2, 2]);
        let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));

        // Define a chain of operations: B = A + 1, C = B * 2, D = C - 3
        // Create scalar constants
        let scalar_one = graph.constant_scalar(1.0f32, MPSDataType::Float32);
        let scalar_two = graph.constant_scalar(2.0f32, MPSDataType::Float32);
        let scalar_three = graph.constant_scalar(3.0f32, MPSDataType::Float32);

        // Define operations using the scalars
        let b = graph.add(&a, &scalar_one, None);
        let c = graph.multiply(&b, &scalar_two, None);
        let d = graph.subtract(&c, &scalar_three, None);

        // Create input tensor with Metal buffer
        let a_tensor = TensorBuffer::new(
            &metal_device,
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            MPSDataType::Float32,
        );

        // Create input feeds
        let mut feeds = HashMap::new();
        feeds.insert(a.clone(), a_tensor.tensor_data.clone());

        // Run graph directly
        let results = graph.run_with_feeds(&feeds, &[b.clone(), c.clone(), d.clone()]);

        // Verify all results contain the expected tensors
        assert!(results.contains_key(&b), "Results should contain tensor B");
        assert!(results.contains_key(&c), "Results should contain tensor C");
        assert!(results.contains_key(&d), "Results should contain tensor D");
    }
}

#[test]
fn test_graph_reuse() {
    // Skip if Metal device not available
    if let Some(metal_device) = metal::Device::system_default() {
        let graph = MPSGraph::new();

        // Create input placeholder
        let shape = MPSShape::from_slice(&[2, 2]);
        let a = graph.placeholder(&shape, MPSDataType::Float32, Some("A"));

        // Define operation: B = A * 2
        let scalar_two = graph.constant_scalar(2.0f32, MPSDataType::Float32);
        let b = graph.multiply(&a, &scalar_two, None);

        // Create input data 1
        let a_tensor1 = TensorBuffer::new(
            &metal_device,
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            MPSDataType::Float32,
        );

        // Create input feeds 1
        let mut feeds1 = HashMap::new();
        feeds1.insert(a.clone(), a_tensor1.tensor_data.clone());

        // First execution
        let results1 = graph.run_with_feeds(&feeds1, &[b.clone()]);

        // Verify result 1 contains the expected tensor
        assert!(results1.contains_key(&b), "Results should contain tensor B");

        // Create input data 2 (different values)
        let a_tensor2 = TensorBuffer::new(
            &metal_device,
            &[5.0f32, 6.0, 7.0, 8.0],
            &[2, 2],
            MPSDataType::Float32,
        );

        // Create input feeds 2
        let mut feeds2 = HashMap::new();
        feeds2.insert(a.clone(), a_tensor2.tensor_data.clone());

        // Second execution (reusing the same graph)
        let results2 = graph.run_with_feeds(&feeds2, &[b.clone()]);

        // Verify result 2 contains the expected tensor
        assert!(results2.contains_key(&b), "Results should contain tensor B");
    }
}
