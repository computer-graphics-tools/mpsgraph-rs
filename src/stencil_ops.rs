use objc::runtime::{Object, Class};
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape};

/// The reduction mode for stencil operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum MPSGraphReductionMode {
    /// Min reduction
    Min = 0,
    /// Max reduction
    Max = 1,
    /// Sum reduction
    Sum = 2,
    /// Product reduction
    Product = 3,
    /// Argument Min reduction
    ArgumentMin = 4,
    /// Argument Max reduction
    ArgumentMax = 5,
}

/// Descriptor for stencil operations
pub struct MPSGraphStencilOpDescriptor(pub(crate) *mut Object);

impl MPSGraphStencilOpDescriptor {
    /// Creates a new stencil operation descriptor with default values
    pub fn new() -> Self {
        unsafe {
            let cls = Class::get("MPSGraphStencilOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphStencilOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified padding style
    pub fn with_padding_style(padding_style: crate::convolution_transpose_ops::PaddingStyle) -> Self {
        unsafe {
            let cls = Class::get("MPSGraphStencilOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptorWithPaddingStyle:padding_style as i64];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphStencilOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified explicit padding
    pub fn with_explicit_padding(explicit_padding: &MPSShape) -> Self {
        unsafe {
            let cls = Class::get("MPSGraphStencilOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptorWithExplicitPadding:explicit_padding.0];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphStencilOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new stencil operation descriptor with the specified offsets and explicit padding
    pub fn with_offsets_and_explicit_padding(offsets: &MPSShape, explicit_padding: &MPSShape) -> Self {
        unsafe {
            let cls = Class::get("MPSGraphStencilOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptorWithOffsets:offsets.0 explicitPadding:explicit_padding.0];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphStencilOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new stencil operation descriptor with all parameters specified
    pub fn with_all_params(
        reduction_mode: MPSGraphReductionMode,
        offsets: &MPSShape,
        strides: &MPSShape,
        dilation_rates: &MPSShape,
        explicit_padding: &MPSShape,
        boundary_mode: crate::sample_grid_ops::MPSGraphPaddingMode,
        padding_style: crate::convolution_transpose_ops::PaddingStyle,
        padding_constant: f32
    ) -> Self {
        unsafe {
            let cls = Class::get("MPSGraphStencilOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, 
                descriptorWithReductionMode:reduction_mode as u64
                offsets:offsets.0
                strides:strides.0
                dilationRates:dilation_rates.0
                explicitPadding:explicit_padding.0
                boundaryMode:boundary_mode as i64
                paddingStyle:padding_style as u64
                paddingConstant:padding_constant
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphStencilOpDescriptor(descriptor)
        }
    }
    
    /// Sets the reduction mode
    pub fn set_reduction_mode(&self, mode: MPSGraphReductionMode) {
        unsafe {
            let _: () = msg_send![self.0, setReductionMode:mode as u64];
        }
    }
    
    /// Sets the offsets
    pub fn set_offsets(&self, offsets: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setOffsets:offsets.0];
        }
    }
    
    /// Sets the strides
    pub fn set_strides(&self, strides: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setStrides:strides.0];
        }
    }
    
    /// Sets the dilation rates
    pub fn set_dilation_rates(&self, dilation_rates: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRates:dilation_rates.0];
        }
    }
    
    /// Sets the explicit padding
    pub fn set_explicit_padding(&self, explicit_padding: &MPSShape) {
        unsafe {
            let _: () = msg_send![self.0, setExplicitPadding:explicit_padding.0];
        }
    }
    
    /// Sets the boundary mode
    pub fn set_boundary_mode(&self, mode: crate::sample_grid_ops::MPSGraphPaddingMode) {
        unsafe {
            let _: () = msg_send![self.0, setBoundaryMode:mode as i64];
        }
    }
    
    /// Sets the padding style
    pub fn set_padding_style(&self, style: crate::convolution_transpose_ops::PaddingStyle) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingStyle:style as u64];
        }
    }
    
    /// Sets the padding constant
    pub fn set_padding_constant(&self, value: f32) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingConstant:value];
        }
    }
}

impl Drop for MPSGraphStencilOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphStencilOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphStencilOpDescriptor(desc)
        }
    }
}

/// Stencil operations for MPSGraph
impl MPSGraph {
    /// Creates a stencil operation and returns the result tensor.
    ///
    /// Performs a weighted reduction operation (See `MPSGraphReductionMode`) on the last 4 dimensions of the `source`
    /// over the window determined by `weights`, according to the value defined in `descriptor`.
    /// The operation can be represented as:
    /// 
    /// `y[i] = reduction{j in w} ( x[i + j] * w[j] )`
    ///
    /// # Arguments
    ///
    /// * `source` - The tensor containing the source data. Must be of rank 4 or greater.
    /// * `weights` - A 4-D tensor containing the weights data.
    /// * `descriptor` - The descriptor object that specifies the parameters for the stencil operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor object.
    pub fn stencil(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        descriptor: &MPSGraphStencilOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                stencilWithSourceTensor:source.0
                weightsTensor:weights.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MPSDataType, tests::should_skip_test};
    use crate::sample_grid_ops::MPSGraphPaddingMode;
    use crate::convolution_transpose_ops::PaddingStyle;
    use std::collections::HashMap;
    
    #[test]
    fn test_stencil_op() {
        if should_skip_test("test_stencil_op") {
            return;
        }
        
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
        feeds.insert(&source, crate::MPSGraphTensorData::from_bytes(&source_data, &source_shape, MPSDataType::Float32));
        feeds.insert(&weights, crate::MPSGraphTensorData::from_bytes(&weights_data, &weights_shape, MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&result]);
        
        // Get the result data
        let result_data = results[&result].to_vec::<f32>();
        
        // With no padding and 3x3 kernel on 5x5 input, we get a 3x3 output
        assert_eq!(result_data.len(), 9);
        
        // Expected values for Sum reduction with 3x3 window of all 1.0 weights:
        // First row: sum of values in the first 3x3 window
        assert_eq!(result_data[0], 1.0 + 2.0 + 3.0 + 6.0 + 7.0 + 8.0 + 11.0 + 12.0 + 13.0);
        // Middle cell: sum of values in the center 3x3 window
        assert_eq!(result_data[4], 7.0 + 8.0 + 9.0 + 12.0 + 13.0 + 14.0 + 17.0 + 18.0 + 19.0);
        // Last cell: sum of values in the bottom-right 3x3 window
        assert_eq!(result_data[8], 13.0 + 14.0 + 15.0 + 18.0 + 19.0 + 20.0 + 23.0 + 24.0 + 25.0);
    }
}