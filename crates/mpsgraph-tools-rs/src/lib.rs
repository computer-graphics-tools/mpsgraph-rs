//! MPSGraph Tools - High-level utilities for MPSGraph
//!
//! This crate provides high-level utilities and ergonomic APIs for working with 
//! Apple's Metal Performance Shaders Graph (MPSGraph) framework through the mpsgraph crate.
//!
//! # Features
//!
//! - **Tensor Operations API**: Ergonomic, functional-style tensor operations with operator overloading
//! - **Utility Functions**: Convenience methods for common tensor operations
//! - **Tensor Creation Helpers**: Easy creation of tensors with different initialization patterns

// Re-export the core mpsgraph types for convenience
pub use mpsgraph::{
    MPSCommandBuffer, MPSDataType, MPSGraph, MPSGraphOperation, MPSGraphTensor,
    MPSGraphTensorData, MPSShape, MPSTensorDataScalar,
};

// Tensor operations module
pub mod tensor_ops;

/// Convenience prelude module with most commonly used items
pub mod prelude {
    // Tensor operations
    pub use crate::tensor_ops;
    pub use crate::tensor_ops::{
        Tensor, GraphExt,
        abs, clip, exp, gelu, log, pow, relu, sigmoid, silu, sqrt, square, tanh,
    };
    
    // Re-export the core mpsgraph types
    pub use mpsgraph::{
        MPSCommandBuffer, MPSDataType, MPSGraph, MPSGraphOperation, MPSGraphTensor,
        MPSGraphTensorData, MPSShape, MPSTensorDataScalar,
    };
}

#[cfg(test)]
mod tests;