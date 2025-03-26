use mpsgraph::{
    MPSGraph, 
    MPSGraphTensorData, 
    MPSDataType, 
    MPSShape,
    MPSGraphConvolution2DOpDescriptor,
    MPSGraphDepthwiseConvolution2DOpDescriptor,
    TensorNamedDataLayout,
    PaddingStyle
};
use std::collections::HashMap;

fn main() {
    // Create a new graph
    let graph = MPSGraph::new();
    
    // === Transposed Convolution (Deconvolution) Example ===
    println!("1. Transposed Convolution (Deconvolution) Example");
    
    // Create a 1x1x4x4 input tensor (NCHW format)
    let input_shape = MPSShape::from_slice(&[1, 1, 4, 4]);
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    ];
    let input_tensor = graph.placeholder(&input_shape, MPSDataType::Float32, None);
    
    // Create a 1x1x3x3 kernel (weights) tensor
    let weights_shape = MPSShape::from_slice(&[1, 1, 3, 3]);
    let weights_data = vec![
        1.0f32, 1.0, 1.0,
        1.0, 2.0, 1.0,
        1.0, 1.0, 1.0
    ];
    let weights_tensor = graph.placeholder(&weights_shape, MPSDataType::Float32, None);
    
    // Define the expected output shape
    let transposed_output_shape = MPSShape::from_slice(&[1, 1, 6, 6]);
    
    // Create descriptor for transposed convolution
    let transposed_descriptor = MPSGraphConvolution2DOpDescriptor::new();
    transposed_descriptor.set_stride_in_x(1);
    transposed_descriptor.set_stride_in_y(1);
    transposed_descriptor.set_padding_style(PaddingStyle::Explicit);
    transposed_descriptor.set_explicit_padding(0, 0, 0, 0);
    transposed_descriptor.set_data_layout(TensorNamedDataLayout::NCHW);
    transposed_descriptor.set_weights_layout(TensorNamedDataLayout::NCHW);
    
    // Create the transposed convolution operation
    let transposed_result = graph.convolution_transpose_2d(
        &input_tensor,
        &weights_tensor,
        &transposed_output_shape,
        &transposed_descriptor,
        Some("transposed_conv")
    );
    
    // === Depthwise Convolution Example ===
    println!("2. Depthwise Convolution Example");
    
    // Create a 1x3x5x5 input tensor with 3 channels (NCHW format)
    let depthwise_input_shape = MPSShape::from_slice(&[1, 3, 5, 5]);
    // Generate some sample data (25 values per channel, 3 channels)
    let mut depthwise_input_data = Vec::with_capacity(75);
    for c in 0..3 {
        for i in 0..25 {
            depthwise_input_data.push((c as f32 + 1.0) * (i as f32 + 1.0));
        }
    }
    let depthwise_input_tensor = graph.placeholder(&depthwise_input_shape, MPSDataType::Float32, None);
    
    // Create a 3x1x3x3 kernel for depthwise convolution (one 3x3 kernel per input channel)
    let depthwise_weights_shape = MPSShape::from_slice(&[3, 1, 3, 3]);
    let depthwise_weights_data = vec![
        // Channel 1 weights
        1.0f32, 1.0, 1.0,
        1.0, 2.0, 1.0,
        1.0, 1.0, 1.0,
        // Channel 2 weights
        0.5f32, 0.5, 0.5,
        0.5, 1.0, 0.5,
        0.5, 0.5, 0.5,
        // Channel 3 weights
        0.3f32, 0.3, 0.3,
        0.3, 0.5, 0.3,
        0.3, 0.3, 0.3
    ];
    let depthwise_weights_tensor = graph.placeholder(&depthwise_weights_shape, MPSDataType::Float32, None);
    
    // Create descriptor for depthwise convolution
    let depthwise_descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor::new_with_layouts(
        TensorNamedDataLayout::NCHW,
        TensorNamedDataLayout::NCHW
    );
    depthwise_descriptor.set_stride_in_x(1);
    depthwise_descriptor.set_stride_in_y(1);
    depthwise_descriptor.set_padding_style(PaddingStyle::Explicit);
    depthwise_descriptor.set_explicit_padding(1, 1, 1, 1);
    
    // Create the depthwise convolution operation
    let depthwise_result = graph.depthwise_convolution_2d(
        &depthwise_input_tensor,
        &depthwise_weights_tensor,
        &depthwise_descriptor,
        Some("depthwise_conv")
    );
    
    // Run the graph with the inputs
    let mut feeds = HashMap::new();
    feeds.insert(&input_tensor, MPSGraphTensorData::new(&input_data, &[1, 1, 4, 4], MPSDataType::Float32));
    feeds.insert(&weights_tensor, MPSGraphTensorData::new(&weights_data, &[1, 1, 3, 3], MPSDataType::Float32));
    feeds.insert(&depthwise_input_tensor, MPSGraphTensorData::new(&depthwise_input_data, &[1, 3, 5, 5], MPSDataType::Float32));
    feeds.insert(&depthwise_weights_tensor, MPSGraphTensorData::new(&depthwise_weights_data, &[3, 1, 3, 3], MPSDataType::Float32));
    
    let results = graph.run(feeds, &[&transposed_result, &depthwise_result]);
    
    // Get and display the results
    let transposed_output = results[&transposed_result].to_vec::<f32>();
    println!("Transposed Convolution Output Shape: 1x1x6x6 (36 elements)");
    println!("Output Size: {}", transposed_output.len());
    println!("First few values: {:.1}, {:.1}, {:.1}, {:.1}, ...", 
             transposed_output[0], transposed_output[1], transposed_output[2], transposed_output[3]);
    
    let depthwise_output = results[&depthwise_result].to_vec::<f32>();
    println!("\nDepthwise Convolution Output Shape: 1x3x5x5 (75 elements)");
    println!("Output Size: {}", depthwise_output.len());
    println!("First few values from each channel:");
    // Print first few values from each channel
    for c in 0..3 {
        let offset = c * 25;
        println!("Channel {}: {:.1}, {:.1}, {:.1}, {:.1}, ...", 
                 c+1, depthwise_output[offset], depthwise_output[offset+1], 
                 depthwise_output[offset+2], depthwise_output[offset+3]);
    }
}