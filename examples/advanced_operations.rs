use metal::Device;
use std::collections::HashMap;

// Import the necessary components from the mpsgraph crate
use mpsgraph::core::{MPSDataType, MPSShape};
use mpsgraph::graph::MPSGraph;
use mpsgraph::tensor_data::MPSGraphTensorData;
use mpsgraph::operation::*;
use mpsgraph::executable::*;

// Import the newly added operation modules
use mpsgraph::stencil_ops::*;
use mpsgraph::fourier_transform_ops::*;
use mpsgraph::non_maximum_suppression_ops::*;
use mpsgraph::quantization_ops::*;
use mpsgraph::call_ops::*;
use mpsgraph::cumulative_ops::*;
use mpsgraph::im2col_ops::*;
use mpsgraph::memory_ops::*;
use mpsgraph::sparse_ops::*;
use mpsgraph::linear_algebra_ops::*;
use mpsgraph::convolution_transpose_ops::*;
use mpsgraph::depthwise_convolution_ops::*;
use mpsgraph::sample_grid_ops::*;

fn main() {
    // Check if a Metal device is available
    if let Some(device) = Device::system_default() {
        println!("Using Metal device: {}", device.name());
        run_advanced_tests();
    } else {
        println!("No Metal device found.");
    }
}

fn run_advanced_tests() {
    println!("Running advanced MPSGraph operations tests on real Metal hardware");
    
    // Test stencil operations
    test_stencil_operations();
    
    // Test Fourier transform operations
    test_fourier_transform();
    
    // Test cumulative operations
    test_cumulative_operations();
    
    // Test linear algebra operations
    test_matrix_inverse();
    
    // Test convolution transpose operations
    test_convolution_transpose();
    
    println!("All advanced tests completed successfully!");
}

fn test_stencil_operations() {
    println!("Testing stencil operations...");
    
    let graph = MPSGraph::new();
    
    // Define a simple 1x5x5x1 input (NHWC)
    let source_shape = MPSShape::from_slice(&[1, 5, 5, 1]);
    let source_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0
    ];
    let source = graph.placeholder(&source_shape, MPSDataType::Float32, None);
    
    // Define a 3x3x1x1 weights - all 1.0 weights will sum the values in a 3x3 window
    let weights_shape = MPSShape::from_slice(&[3, 3, 1, 1]);
    let weights_data = vec![
        1.0f32, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    ];
    let weights = graph.placeholder(&weights_shape, MPSDataType::Float32, None);
    
    // Create descriptor with sum reduction and explicit padding
    let offsets = MPSShape::from_slice(&[0, 0, 0, 0]);
    let strides = MPSShape::from_slice(&[1, 1, 1, 1]);
    let dilation_rates = MPSShape::from_slice(&[1, 1, 1, 1]);
    let explicit_padding = MPSShape::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    
    let descriptor = MPSGraphStencilOpDescriptor::with_all_params(
        MPSGraphReductionMode::Sum,
        &offsets,
        &strides,
        &dilation_rates,
        &explicit_padding,
        MPSGraphPaddingMode::Zero,
        PaddingStyle::Explicit,
        0.0
    );
    
    // Create the stencil operation
    let result = graph.stencil(
        &source,
        &weights,
        &descriptor,
        Some("stencil_op")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&source, MPSGraphTensorData::from_bytes(&source_data, &source_shape, MPSDataType::Float32));
    feeds.insert(&weights, MPSGraphTensorData::from_bytes(&weights_data, &weights_shape, MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&result]);
    
    // Get the result data
    let result_data = results[&result].to_vec::<f32>();
    
    // With no padding and 3x3 kernel on 5x5 input, we get a 3x3 output
    assert_eq!(result_data.len(), 9);
    
    println!("Stencil operation result: {:?}", result_data);
    
    // Expected values for Sum reduction with 3x3 window of all 1.0 weights:
    // First row: sum of values in the first 3x3 window
    let expected_first = 1.0 + 2.0 + 3.0 + 6.0 + 7.0 + 8.0 + 11.0 + 12.0 + 13.0;
    assert!((result_data[0] - expected_first).abs() < 1e-5);
    
    // Middle cell: sum of values in the center 3x3 window
    let expected_middle = 7.0 + 8.0 + 9.0 + 12.0 + 13.0 + 14.0 + 17.0 + 18.0 + 19.0;
    assert!((result_data[4] - expected_middle).abs() < 1e-5);
    
    // Last cell: sum of values in the bottom-right 3x3 window
    let expected_last = 13.0 + 14.0 + 15.0 + 18.0 + 19.0 + 20.0 + 23.0 + 24.0 + 25.0;
    assert!((result_data[8] - expected_last).abs() < 1e-5);
    
    println!("Stencil operations test passed!");
}

fn test_fourier_transform() {
    println!("Testing Fourier transform operations...");
    
    let graph = MPSGraph::new();
    
    // Create a simple 1D signal
    let shape = MPSShape::from_slice(&[8]);
    
    // Simple cosine signal
    let signal_data: Vec<f32> = (0..8).map(|i| (2.0 * std::f32::consts::PI * i as f32 / 8.0).cos()).collect();
    println!("Input signal: {:?}", signal_data);
    
    let signal = graph.placeholder(&shape, MPSDataType::Float32, Some("signal"));
    
    // Create FFT descriptor
    let descriptor = MPSGraphFFTDescriptor::new();
    descriptor.set_scaling_mode(MPSGraphFFTScalingMode::None);
    
    // Calculate FFT for the signal
    let fft_result = graph.fast_fourier_transform(
        &signal,
        &[0], // Transform along the 0th axis
        &descriptor,
        Some("fft_result")
    );
    
    // Calculate inverse FFT to get back the original signal
    let ifft_result = graph.hermitean_to_real_fft(
        &fft_result,
        &[0], // Transform along the 0th axis
        &descriptor,
        Some("ifft_result")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&signal, MPSGraphTensorData::from_bytes(&signal_data, &shape, MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&fft_result, &ifft_result]);
    
    // Get the result data
    let fft_data = results[&fft_result].to_vec::<f32>();
    let ifft_data = results[&ifft_result].to_vec::<f32>();
    
    println!("FFT result: {:?}", fft_data);
    println!("IFFT result: {:?}", ifft_data);
    
    // The IFFT should reconstruct the original signal (within numerical precision)
    for (original, reconstructed) in signal_data.iter().zip(ifft_data.iter()) {
        assert!((original - reconstructed).abs() < 1e-5);
    }
    
    println!("Fourier transform operations test passed!");
}

fn test_cumulative_operations() {
    println!("Testing cumulative operations...");
    
    let graph = MPSGraph::new();
    
    // Create a 2x3 tensor
    let shape = MPSShape::from_slice(&[2, 3]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    let tensor = graph.placeholder(&shape, MPSDataType::Float32, Some("input"));
    
    // Compute cumulative sum along columns (axis 0)
    let cum_sum_0 = graph.cumulative_sum(
        &tensor, 
        0,     // axis = 0 (along columns)
        false, // exclusive = false (inclusive)
        false, // reverse = false (forward)
        Some("cum_sum_0")
    );
    
    // Compute cumulative sum along rows (axis 1)
    let cum_sum_1 = graph.cumulative_sum(
        &tensor, 
        1,     // axis = 1 (along rows)
        false, // exclusive = false (inclusive)
        false, // reverse = false (forward)
        Some("cum_sum_1")
    );
    
    // Compute cumulative product along columns (axis 0)
    let cum_prod_0 = graph.cumulative_product(
        &tensor, 
        0,     // axis = 0 (along columns)
        false, // exclusive = false (inclusive)
        false, // reverse = false (forward)
        Some("cum_prod_0")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&tensor, MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&cum_sum_0, &cum_sum_1, &cum_prod_0]);
    
    // Get the result data
    let cum_sum_0_data = results[&cum_sum_0].to_vec::<f32>();
    let cum_sum_1_data = results[&cum_sum_1].to_vec::<f32>();
    let cum_prod_0_data = results[&cum_prod_0].to_vec::<f32>();
    
    println!("Cumulative sum along axis 0: {:?}", cum_sum_0_data);
    println!("Cumulative sum along axis 1: {:?}", cum_sum_1_data);
    println!("Cumulative product along axis 0: {:?}", cum_prod_0_data);
    
    // Expected results for cumulative sum along axis 0:
    // [1, 2, 3]
    // [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_eq!(cum_sum_0_data, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    
    // Expected results for cumulative sum along axis 1:
    // [1, 1+2, 1+2+3] = [1, 3, 6]
    // [4, 4+5, 4+5+6] = [4, 9, 15]
    assert_eq!(cum_sum_1_data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    
    // Expected results for cumulative product along axis 0:
    // [1, 2, 3]
    // [1*4, 2*5, 3*6] = [4, 10, 18]
    assert_eq!(cum_prod_0_data, vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]);
    
    println!("Cumulative operations test passed!");
}

fn test_matrix_inverse() {
    println!("Testing matrix inverse operation...");
    
    let graph = MPSGraph::new();
    
    // Create a 2x2 invertible matrix
    let shape = MPSShape::from_slice(&[2, 2]);
    let data = vec![4.0f32, 7.0, 2.0, 6.0]; // Matrix [[4, 7], [2, 6]]
    
    let matrix = graph.placeholder(&shape, MPSDataType::Float32, Some("matrix"));
    
    // Compute the inverse of the matrix
    let inverse = graph.matrix_inverse(&matrix, Some("inverse"));
    
    // Multiply original matrix by its inverse to get identity matrix
    let identity = graph.matmul(&matrix, &inverse, Some("identity"));
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&matrix, MPSGraphTensorData::from_bytes(&data, &shape, MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&inverse, &identity]);
    
    // Get the result data
    let inverse_data = results[&inverse].to_vec::<f32>();
    let identity_data = results[&identity].to_vec::<f32>();
    
    println!("Original matrix: {:?}", data);
    println!("Inverse matrix: {:?}", inverse_data);
    println!("Identity check (matrix * inverse): {:?}", identity_data);
    
    // Check the result - M * M^-1 should be approximately the identity matrix
    assert!((identity_data[0] - 1.0).abs() < 1e-5); // [0,0] ≈ 1
    assert!(identity_data[1].abs() < 1e-5);          // [0,1] ≈ 0
    assert!(identity_data[2].abs() < 1e-5);          // [1,0] ≈ 0
    assert!((identity_data[3] - 1.0).abs() < 1e-5); // [1,1] ≈ 1
    
    println!("Matrix inverse operation test passed!");
}

fn test_convolution_transpose() {
    println!("Testing convolution transpose operation...");
    
    let graph = MPSGraph::new();
    
    // Create a 1x3x3x1 input tensor (NCHW format)
    let input_shape = MPSShape::from_slice(&[1, 1, 3, 3]);
    let input_data = vec![
        1.0f32, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ];
    
    // Create a 1x1x2x2 weight tensor
    let weights_shape = MPSShape::from_slice(&[1, 1, 2, 2]);
    let weights_data = vec![
        1.0f32, 1.0,
        1.0, 1.0
    ];
    
    let input = graph.placeholder(&input_shape, MPSDataType::Float32, Some("input"));
    let weights = graph.placeholder(&weights_shape, MPSDataType::Float32, Some("weights"));
    
    // Create a 1x6x6 output shape tensor
    let output_shape = MPSShape::from_slice(&[1, 1, 6, 6]);
    
    // Create the convolution transpose operation
    let result = graph.convolution_transpose2d(
        &input,
        &weights,
        &graph.constant_from_bytes(&[1i32, 1, 6, 6], &MPSShape::from_slice(&[4]), MPSDataType::Int32, None),
        (2, 2), // strides
        (0, 0, 0, 0), // padding (left, right, top, bottom)
        (1, 1), // dilations
        Some("conv_transpose")
    );
    
    // Run the graph
    let mut feeds = HashMap::new();
    feeds.insert(&input, MPSGraphTensorData::from_bytes(&input_data, &input_shape, MPSDataType::Float32));
    feeds.insert(&weights, MPSGraphTensorData::from_bytes(&weights_data, &weights_shape, MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&result]);
    
    // Get the result data
    let result_data = results[&result].to_vec::<f32>();
    
    println!("Convolution transpose result: {:?}", result_data);
    println!("Result shape: {:?}", results[&result].shape().dimensions());
    
    // The result should have shape [1, 1, 6, 6] for a 3x3 input, 2x2 kernel, and stride 2
    assert_eq!(results[&result].shape().dimensions(), vec![1, 1, 6, 6]);
    
    println!("Convolution transpose operation test passed!");
}