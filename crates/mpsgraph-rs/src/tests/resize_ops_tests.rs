use crate::{
    convolution_transpose_ops::TensorNamedDataLayout,
    core::MPSDataType,
    graph::MPSGraph,
    resize_ops::{MPSGraphResizeMode, MPSGraphResizeNearestRoundingMode},
    shape::MPSShape,
    tensor_data::MPSGraphTensorData,
    tensor::MPSGraphTensor,
};
use std::collections::HashMap;

// Simple verification helper that checks if result exists
fn verify_result_exists(results: &HashMap<MPSGraphTensor, MPSGraphTensorData>, tensor: &MPSGraphTensor, name: &str) {
    assert!(results.contains_key(tensor), "Results should contain {} tensor", name);
}

#[test]
fn test_resize_basic() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("Input"));
    
    // Create target size for resizing
    let target_size = MPSShape::from_slice(&[32, 32]);
    
    // Test different resize modes
    let resize_nearest = graph.resize(
        &input,
        &target_size,
        MPSGraphResizeMode::Nearest,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeNearest"),
    );
    
    let resize_bilinear = graph.resize(
        &input,
        &target_size,
        MPSGraphResizeMode::Bilinear,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeBilinear"),
    );
    
    // Create input data
    let mut data = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        data.push(i as f32);
    }
    
    let input_data = MPSGraphTensorData::new(&data, &[1, 3, 16, 16], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[
        resize_nearest.clone(),
        resize_bilinear.clone(),
    ]);
    
    // Verify results
    verify_result_exists(&results, &resize_nearest, "resize_nearest");
    verify_result_exists(&results, &resize_bilinear, "resize_bilinear");
}

#[test]
fn test_resize_with_size_tensor() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("Input"));
    
    // Create size tensor
    let size_data = [32i64, 32];
    let size_tensor = graph.constant(&size_data, &MPSShape::from_slice(&[2]), MPSDataType::Int64);
    
    // Test resize with size tensor
    let resize = graph.resize_with_size_tensor(
        &input,
        &size_tensor,
        MPSGraphResizeMode::Bilinear,
        true, // center_result
        true, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeWithSizeTensor"),
    );
    
    // Create input data
    let mut data = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        data.push(i as f32);
    }
    
    let input_data = MPSGraphTensorData::new(&data, &[1, 3, 16, 16], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[resize.clone()]);
    
    // Verify results
    verify_result_exists(&results, &resize, "resize_with_size_tensor");
}

#[test]
fn test_resize_nearest_with_rounding_modes() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("Input"));
    
    // Create size tensor
    let size_data = [24i64, 24];
    let size_tensor = graph.constant(&size_data, &MPSShape::from_slice(&[2]), MPSDataType::Int64);
    
    // Test resize with different rounding modes
    let resize_ceil = graph.resize_nearest(
        &input,
        &size_tensor,
        MPSGraphResizeNearestRoundingMode::Ceil,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeNearestCeil"),
    );
    
    let resize_floor = graph.resize_nearest(
        &input,
        &size_tensor,
        MPSGraphResizeNearestRoundingMode::Floor,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeNearestFloor"),
    );
    
    let resize_round_prefer_ceil = graph.resize_nearest(
        &input,
        &size_tensor,
        MPSGraphResizeNearestRoundingMode::RoundPreferCeil,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeNearestRoundPreferCeil"),
    );
    
    // Create input data
    let mut data = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        data.push(i as f32);
    }
    
    let input_data = MPSGraphTensorData::new(&data, &[1, 3, 16, 16], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[
        resize_ceil.clone(),
        resize_floor.clone(),
        resize_round_prefer_ceil.clone(),
    ]);
    
    // Verify results
    verify_result_exists(&results, &resize_ceil, "resize_ceil");
    verify_result_exists(&results, &resize_floor, "resize_floor");
    verify_result_exists(&results, &resize_round_prefer_ceil, "resize_round_prefer_ceil");
}

#[test]
fn test_resize_bilinear_specific() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("Input"));
    
    // Create size tensor
    let size_data = [32i64, 32];
    let size_tensor = graph.constant(&size_data, &MPSShape::from_slice(&[2]), MPSDataType::Int64);
    
    // Test specific bilinear resize function
    let resize_bilinear = graph.resize_bilinear(
        &input,
        &size_tensor,
        true, // center_result
        true, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeBilinearSpecific"),
    );
    
    // Create input data
    let mut data = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        data.push(i as f32);
    }
    
    let input_data = MPSGraphTensorData::new(&data, &[1, 3, 16, 16], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[resize_bilinear.clone()]);
    
    // Verify results
    verify_result_exists(&results, &resize_bilinear, "resize_bilinear_specific");
}

#[test]
fn test_resize_with_scale_offset() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&shape, MPSDataType::Float32, Some("Input"));
    
    // Create size tensor
    let size_data = [32i64, 32];
    let size_tensor = graph.constant(&size_data, &MPSShape::from_slice(&[2]), MPSDataType::Int64);
    
    // Create scale-offset tensor [scaleY, scaleX, offsetY, offsetX]
    let scale_offset_data = [0.5f32, 0.5, 0.0, 0.0];
    let scale_offset_tensor = graph.constant(
        &scale_offset_data,
        &MPSShape::from_slice(&[4]),
        MPSDataType::Float32,
    );
    
    // Test resize with scale-offset
    let resize_scale_offset = graph.resize_with_scale_offset(
        &input,
        &size_tensor,
        &scale_offset_tensor,
        MPSGraphResizeMode::Bilinear,
        TensorNamedDataLayout::NCHW,
        Some("ResizeWithScaleOffset"),
    );
    
    // Create input data
    let mut data = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        data.push(i as f32);
    }
    
    let input_data = MPSGraphTensorData::new(&data, &[1, 3, 16, 16], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[resize_scale_offset.clone()]);
    
    // Verify results
    verify_result_exists(&results, &resize_scale_offset, "resize_scale_offset");
}

#[test]
fn test_resize_gradient() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let input_shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&input_shape, MPSDataType::Float32, Some("Input"));
    
    // Create target size for resizing
    let target_size = MPSShape::from_slice(&[32, 32]);
    
    // Forward resize operation
    let resized = graph.resize(
        &input,
        &target_size,
        MPSGraphResizeMode::Bilinear,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("Resize"),
    );
    
    // Create gradient tensor (matching the shape of the resized output)
    let gradient_shape = MPSShape::from_slice(&[1, 3, 32, 32]); // Same as resized output
    let gradient = graph.placeholder(&gradient_shape, MPSDataType::Float32, Some("Gradient"));
    
    // Compute resize gradient
    let resize_grad = graph.resize_gradient(
        &gradient,
        &input,
        MPSGraphResizeMode::Bilinear,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeGradient"),
    );
    
    // Create specific bilinear gradient
    let resize_bilinear_grad = graph.resize_bilinear_gradient(
        &gradient,
        &input,
        true, // center_result
        false, // align_corners
        TensorNamedDataLayout::NCHW,
        Some("ResizeBilinearGradient"),
    );
    
    // Create input data
    let mut input_data_vec = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        input_data_vec.push(i as f32);
    }
    
    // Create gradient data (all ones for simplicity)
    let mut gradient_data_vec = Vec::with_capacity(1 * 3 * 32 * 32);
    for _ in 0..(1 * 3 * 32 * 32) {
        gradient_data_vec.push(1.0f32);
    }
    
    let input_data = MPSGraphTensorData::new(&input_data_vec, &[1, 3, 16, 16], MPSDataType::Float32);
    let gradient_data = MPSGraphTensorData::new(&gradient_data_vec, &[1, 3, 32, 32], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    feeds.insert(gradient.clone(), gradient_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[
        resized.clone(),
        resize_grad.clone(),
        resize_bilinear_grad.clone(),
    ]);
    
    // Verify results
    verify_result_exists(&results, &resized, "resized");
    verify_result_exists(&results, &resize_grad, "resize_grad");
    verify_result_exists(&results, &resize_bilinear_grad, "resize_bilinear_grad");
}

#[test]
fn test_resize_gradient_with_scale_offset() {
    let graph = MPSGraph::new();
    
    // Create input tensor (NCHW: 1, 3, 16, 16)
    let input_shape = MPSShape::from_slice(&[1, 3, 16, 16]);
    let input = graph.placeholder(&input_shape, MPSDataType::Float32, Some("Input"));
    
    // Create size tensor
    let size_data = [32i64, 32];
    let size_tensor = graph.constant(&size_data, &MPSShape::from_slice(&[2]), MPSDataType::Int64);
    
    // Create scale-offset tensor [scaleY, scaleX, offsetY, offsetX]
    let scale_offset_data = [0.5f32, 0.5, 0.0, 0.0];
    let scale_offset_tensor = graph.constant(
        &scale_offset_data,
        &MPSShape::from_slice(&[4]),
        MPSDataType::Float32,
    );
    
    // Forward resize operation with scale-offset
    let resized = graph.resize_with_scale_offset(
        &input,
        &size_tensor,
        &scale_offset_tensor,
        MPSGraphResizeMode::Bilinear,
        TensorNamedDataLayout::NCHW,
        Some("ResizeWithScaleOffset"),
    );
    
    // Create gradient tensor (matching the shape of the resized output)
    let gradient_shape = MPSShape::from_slice(&[1, 3, 32, 32]); // Same as resized output
    let gradient = graph.placeholder(&gradient_shape, MPSDataType::Float32, Some("Gradient"));
    
    // Compute resize gradient with scale-offset
    let resize_grad_scale_offset = graph.resize_gradient_with_scale_offset(
        &gradient,
        &input,
        &scale_offset_tensor,
        MPSGraphResizeMode::Bilinear,
        TensorNamedDataLayout::NCHW,
        Some("ResizeGradientWithScaleOffset"),
    );
    
    // Compute bilinear resize gradient with scale-offset
    let resize_bilinear_grad_scale_offset = graph.resize_bilinear_gradient_with_scale_offset(
        &gradient,
        &input,
        &scale_offset_tensor,
        TensorNamedDataLayout::NCHW,
        Some("ResizeBilinearGradientWithScaleOffset"),
    );
    
    // Create input data
    let mut input_data_vec = Vec::with_capacity(1 * 3 * 16 * 16);
    for i in 0..(1 * 3 * 16 * 16) {
        input_data_vec.push(i as f32);
    }
    
    // Create gradient data (all ones for simplicity)
    let mut gradient_data_vec = Vec::with_capacity(1 * 3 * 32 * 32);
    for _ in 0..(1 * 3 * 32 * 32) {
        gradient_data_vec.push(1.0f32);
    }
    
    let input_data = MPSGraphTensorData::new(&input_data_vec, &[1, 3, 16, 16], MPSDataType::Float32);
    let gradient_data = MPSGraphTensorData::new(&gradient_data_vec, &[1, 3, 32, 32], MPSDataType::Float32);
    
    // Create feeds
    let mut feeds = HashMap::new();
    feeds.insert(input.clone(), input_data);
    feeds.insert(gradient.clone(), gradient_data);
    
    // Run graph
    let results = graph.run_with_feeds(&feeds, &[
        resized.clone(),
        resize_grad_scale_offset.clone(),
        resize_bilinear_grad_scale_offset.clone(),
    ]);
    
    // Verify results
    verify_result_exists(&results, &resized, "resized");
    verify_result_exists(&results, &resize_grad_scale_offset, "resize_grad_scale_offset");
    verify_result_exists(&results, &resize_bilinear_grad_scale_offset, "resize_bilinear_grad_scale_offset");
}