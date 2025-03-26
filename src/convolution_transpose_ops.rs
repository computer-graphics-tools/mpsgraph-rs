use objc::runtime::Object;
use objc::msg_send;
use objc::runtime::{YES, NO};
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape};

/// Defines the data layout for tensors
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum TensorNamedDataLayout {
    NCHW = 0, // Batch, Channels, Height, Width
    NHWC = 1, // Batch, Height, Width, Channels
    CHWN = 2, // Channels, Height, Width, Batch
    HWC = 3,  // Height, Width, Channels
    CHW = 4,  // Channels, Height, Width
}

/// Defines the padding style for convolution operations
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum PaddingStyle {
    Explicit = 0,
    TfSame = 1,
    TfValid = 2,
}

/// Descriptor for 2D convolution operations
pub struct MPSGraphConvolution2DOpDescriptor(pub(crate) *mut Object);

impl MPSGraphConvolution2DOpDescriptor {
    /// Creates a new convolution 2D operation descriptor
    pub fn new() -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphConvolution2DOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphConvolution2DOpDescriptor(descriptor)
        }
    }
    
    /// Sets the stride in X dimension
    pub fn set_stride_in_x(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInX:stride];
        }
    }
    
    /// Sets the stride in Y dimension
    pub fn set_stride_in_y(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self.0, setStrideInY:stride];
        }
    }
    
    /// Sets the dilation rate in X dimension
    pub fn set_dilation_rate_in_x(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInX:rate];
        }
    }
    
    /// Sets the dilation rate in Y dimension
    pub fn set_dilation_rate_in_y(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self.0, setDilationRateInY:rate];
        }
    }
    
    /// Sets the padding on the left
    pub fn set_padding_left(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingLeft:padding];
        }
    }
    
    /// Sets the padding on the right
    pub fn set_padding_right(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingRight:padding];
        }
    }
    
    /// Sets the padding on the top
    pub fn set_padding_top(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingTop:padding];
        }
    }
    
    /// Sets the padding on the bottom
    pub fn set_padding_bottom(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingBottom:padding];
        }
    }
    
    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingStyle) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingStyle:style as u64];
        }
    }
    
    /// Sets the data layout
    pub fn set_data_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self.0, setDataLayout:layout as u64];
        }
    }
    
    /// Sets the weights layout
    pub fn set_weights_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self.0, setWeightsLayout:layout as u64];
        }
    }
    
    /// Sets the explicit padding values
    pub fn set_explicit_padding(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            let _: () = msg_send![
                self.0,
                setExplicitPaddingWithPaddingLeft:left
                paddingRight:right
                paddingTop:top
                paddingBottom:bottom
            ];
        }
    }
    
    /// Sets whether to use same padding in the TensorFlow style
    pub fn set_use_same_padding(&self, use_same_padding: bool) {
        unsafe {
            let _: () = msg_send![self.0, setUseSamePadding:if use_same_padding { YES } else { NO }];
        }
    }
}

impl Drop for MPSGraphConvolution2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphConvolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphConvolution2DOpDescriptor(desc)
        }
    }
}

/// Transposed convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D convolution transpose operation and returns the result tensor.
    ///
    /// Convolution Tranpose operation is exactly the same as convolution gradient with respect to input image.
    /// Weights tensor and source tensors are interpreted as they are in convolution data gradient.
    /// Convolution with stride `s` downsamples source tensor by factor `s` in spatial dimensions whereas
    /// convolution tranpose with stride `s` upsamples source tensor by factor `s`.
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape` - Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn convolution_transpose_2d(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DWithSourceTensor:source.0
                weightsTensor:weights.0
                outputShape:output_shape.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D convolution transpose operation with tensor output shape
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn convolution_transpose_2d_with_tensor_shape(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DWithSourceTensor:source.0
                weightsTensor:weights.0
                outputShapeTensor:output_shape_tensor.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a convolution transpose gradient operation with respect to the source tensor
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape` - Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to source
    pub fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor:incoming_gradient.0
                weightsTensor:weights.0
                outputShape:output_shape.0
                forwardConvolutionDescriptor:forward_convolution_descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a convolution transpose gradient operation with respect to the source tensor (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to source
    pub fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor:incoming_gradient.0
                weightsTensor:weights.0
                outputShapeTensor:output_shape_tensor.0
                forwardConvolutionDescriptor:forward_convolution_descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a convolution transpose weights gradient operation
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape` - Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape: &MPSShape,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor:incoming_gradient.0
                sourceTensor:source.0
                outputShape:output_shape.0
                forwardConvolutionDescriptor:forward_convolution_descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a convolution transpose weights gradient operation (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape_tensor: &MPSGraphTensor,
        forward_convolution_descriptor: &MPSGraphConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor:incoming_gradient.0
                sourceTensor:source.0
                outputShapeTensor:output_shape_tensor.0
                forwardConvolutionDescriptor:forward_convolution_descriptor.0
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
    use std::collections::HashMap;
    
    #[test]
    fn test_convolution_transpose_2d() {
        if should_skip_test("test_convolution_transpose_2d") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Define a simple 1x1x4x4 input (NCHW)
        let source_shape = MPSShape::from_slice(&[1, 1, 4, 4]);
        let source_data = vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ];
        let source = graph.placeholder(&source_shape, MPSDataType::Float32, None);
        
        // Define a 1x1x3x3 kernel (OIHW)
        let weights_shape = MPSShape::from_slice(&[1, 1, 3, 3]);
        let weights_data = vec![
            1.0f32, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ];
        let weights = graph.placeholder(&weights_shape, MPSDataType::Float32, None);
        
        // Define the expected output shape
        let output_shape = MPSShape::from_slice(&[1, 1, 6, 6]);
        
        // Create descriptor with stride 1, no dilation, explicit padding
        let descriptor = MPSGraphConvolution2DOpDescriptor::new();
        descriptor.set_stride_in_x(1);
        descriptor.set_stride_in_y(1);
        descriptor.set_dilation_rate_in_x(1);
        descriptor.set_dilation_rate_in_y(1);
        descriptor.set_explicit_padding(0, 0, 0, 0);
        descriptor.set_padding_style(PaddingStyle::Explicit);
        descriptor.set_data_layout(TensorNamedDataLayout::NCHW);
        descriptor.set_weights_layout(TensorNamedDataLayout::NCHW);
        
        // Create the transpose convolution operation
        let result = graph.convolution_transpose_2d(
            &source,
            &weights,
            &output_shape,
            &descriptor,
            Some("transpose_conv")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&source, crate::MPSGraphTensorData::new(&source_data, &[1, 1, 4, 4], MPSDataType::Float32));
        feeds.insert(&weights, crate::MPSGraphTensorData::new(&weights_data, &[1, 1, 3, 3], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&result]);
        
        // Get the result and verify shape
        let result_data = results[&result].to_vec::<f32>();
        assert_eq!(result_data.len(), 36); // 6x6=36 elements expected
    }
}