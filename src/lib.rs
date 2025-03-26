#[macro_use]
extern crate objc_rs as objc;

// Core modules
pub mod core;
pub mod device;
pub mod graph;
pub mod tensor;
pub mod tensor_data;
pub mod operation;
pub mod executable;

// Operation-specific modules
pub mod activation_ops;
pub mod arithmetic_ops;
pub mod reduction_ops;
pub mod tensor_shape_ops;
pub mod matrix_ops;
pub mod convolution_ops;
pub mod convolution_transpose_ops;
pub mod depthwise_convolution_ops;
pub mod normalization_ops;
pub mod pooling_ops;
pub mod gradient_ops;
pub mod random_ops;
pub mod loss_ops;
pub mod optimizer_ops;
pub mod rnn_ops;
// Temporarily disable control_flow_ops due to Block_copy issues
// pub mod control_flow_ops;
pub mod linear_algebra_ops;

// Re-export most commonly used types
pub use core::{MPSDataType, MPSShape, MPSGraphOptions, MPSGraphOptimization, MPSGraphOptimizationProfile};
pub use device::MPSGraphDevice;
pub use graph::MPSGraph;
pub use tensor::MPSGraphTensor;
pub use tensor_data::MPSGraphTensorData;
pub use operation::MPSGraphOperation;
pub use executable::MPSGraphExecutable;
pub use random_ops::{MPSGraphRandomDistribution, MPSGraphRandomNormalSamplingMethod, MPSGraphRandomOpDescriptor};
pub use loss_ops::MPSGraphLossReductionType;
pub use rnn_ops::{MPSGraphRNNActivation, MPSGraphSingleGateRNNDescriptor, MPSGraphLSTMDescriptor, MPSGraphGRUDescriptor};
pub use convolution_transpose_ops::{MPSGraphConvolution2DOpDescriptor, TensorNamedDataLayout, PaddingStyle};
pub use depthwise_convolution_ops::{MPSGraphDepthwiseConvolution2DOpDescriptor, MPSGraphDepthwiseConvolution3DOpDescriptor};

/// Convenience prelude module with most commonly used items
pub mod prelude {
    pub use crate::core::{MPSDataType, MPSShape, MPSGraphOptions, MPSGraphOptimization};
    pub use crate::device::MPSGraphDevice;
    pub use crate::graph::MPSGraph;
    pub use crate::tensor::MPSGraphTensor;
    pub use crate::tensor_data::MPSGraphTensorData;
    pub use crate::operation::MPSGraphOperation;
    pub use crate::executable::MPSGraphExecutable;
    pub use crate::random_ops::{MPSGraphRandomDistribution, MPSGraphRandomNormalSamplingMethod, MPSGraphRandomOpDescriptor};
    pub use crate::loss_ops::MPSGraphLossReductionType;
    pub use crate::rnn_ops::{MPSGraphRNNActivation, MPSGraphSingleGateRNNDescriptor, MPSGraphLSTMDescriptor, MPSGraphGRUDescriptor};
    pub use crate::convolution_transpose_ops::{MPSGraphConvolution2DOpDescriptor, TensorNamedDataLayout, PaddingStyle};
    pub use crate::depthwise_convolution_ops::{MPSGraphDepthwiseConvolution2DOpDescriptor, MPSGraphDepthwiseConvolution3DOpDescriptor};
}

#[cfg(test)]
mod tests;