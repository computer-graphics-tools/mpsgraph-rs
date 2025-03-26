use objc::runtime::Object;
use objc::msg_send;
use crate::graph::MPSGraph;
use crate::tensor::MPSGraphTensor;
use crate::core::{NSString, MPSShape};
use crate::convolution_transpose_ops::{TensorNamedDataLayout, PaddingStyle};

/// Descriptor for 2D depthwise convolution operations
pub struct MPSGraphDepthwiseConvolution2DOpDescriptor(pub(crate) *mut Object);

impl MPSGraphDepthwiseConvolution2DOpDescriptor {
    /// Creates a new depthwise convolution 2D operation descriptor
    pub fn new() -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphDepthwiseConvolution2DOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptor];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphDepthwiseConvolution2DOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new depthwise convolution 2D operation descriptor with specified data and weights layouts
    pub fn new_with_layouts(data_layout: TensorNamedDataLayout, weights_layout: TensorNamedDataLayout) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphDepthwiseConvolution2DOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![
                cls, 
                descriptorWithDataLayout:data_layout as u64
                weightsLayout:weights_layout as u64
            ];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphDepthwiseConvolution2DOpDescriptor(descriptor)
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
}

impl Drop for MPSGraphDepthwiseConvolution2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphDepthwiseConvolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphDepthwiseConvolution2DOpDescriptor(desc)
        }
    }
}

/// Descriptor for 3D depthwise convolution operations
pub struct MPSGraphDepthwiseConvolution3DOpDescriptor(pub(crate) *mut Object);

impl MPSGraphDepthwiseConvolution3DOpDescriptor {
    /// Creates a new depthwise convolution 3D operation descriptor with default values
    pub fn new(padding_style: PaddingStyle) -> Self {
        unsafe {
            let cls = objc::runtime::Class::get("MPSGraphDepthwiseConvolution3DOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![cls, descriptorWithPaddingStyle:padding_style as u64];
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphDepthwiseConvolution3DOpDescriptor(descriptor)
        }
    }
    
    /// Creates a new depthwise convolution 3D operation descriptor with given values
    pub fn new_with_values(
        strides: &[usize],
        dilation_rates: &[usize],
        padding_values: &[usize],
        padding_style: PaddingStyle
    ) -> Self {
        unsafe {
            // Convert Rust arrays to NSArrays
            let strides_array = create_number_array(strides);
            let dilation_rates_array = create_number_array(dilation_rates);
            let padding_values_array = create_number_array(padding_values);
            
            let cls = objc::runtime::Class::get("MPSGraphDepthwiseConvolution3DOpDescriptor").unwrap();
            let descriptor: *mut Object = msg_send![
                cls, 
                descriptorWithStrides:strides_array
                dilationRates:dilation_rates_array
                paddingValues:padding_values_array
                paddingStyle:padding_style as u64
            ];
            
            // Release the NSArrays since they're retained by the descriptor
            let _: () = msg_send![strides_array, release];
            let _: () = msg_send![dilation_rates_array, release];
            let _: () = msg_send![padding_values_array, release];
            
            let descriptor: *mut Object = msg_send![descriptor, retain];
            MPSGraphDepthwiseConvolution3DOpDescriptor(descriptor)
        }
    }
    
    /// Sets the strides for spatial dimensions (3 values)
    pub fn set_strides(&self, strides: &[usize]) {
        unsafe {
            let strides_array = create_number_array(strides);
            let _: () = msg_send![self.0, setStrides:strides_array];
            let _: () = msg_send![strides_array, release];
        }
    }
    
    /// Sets the dilation rates for spatial dimensions (3 values)
    pub fn set_dilation_rates(&self, dilation_rates: &[usize]) {
        unsafe {
            let dilation_rates_array = create_number_array(dilation_rates);
            let _: () = msg_send![self.0, setDilationRates:dilation_rates_array];
            let _: () = msg_send![dilation_rates_array, release];
        }
    }
    
    /// Sets the padding values for spatial dimensions (6 values)
    pub fn set_padding_values(&self, padding_values: &[usize]) {
        unsafe {
            let padding_values_array = create_number_array(padding_values);
            let _: () = msg_send![self.0, setPaddingValues:padding_values_array];
            let _: () = msg_send![padding_values_array, release];
        }
    }
    
    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingStyle) {
        unsafe {
            let _: () = msg_send![self.0, setPaddingStyle:style as u64];
        }
    }
    
    /// Sets the channel dimension index
    pub fn set_channel_dimension_index(&self, index: isize) {
        unsafe {
            let _: () = msg_send![self.0, setChannelDimensionIndex:index];
        }
    }
}

impl Drop for MPSGraphDepthwiseConvolution3DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

impl Clone for MPSGraphDepthwiseConvolution3DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let desc: *mut Object = msg_send![self.0, copy];
            MPSGraphDepthwiseConvolution3DOpDescriptor(desc)
        }
    }
}

/// Helper function to create NSArray of NSNumber objects from a slice of usizes
fn create_number_array(values: &[usize]) -> *mut Object {
    unsafe {
        let cls = objc::runtime::Class::get("NSNumber").unwrap();
        let numbers: Vec<*mut Object> = values.iter()
            .map(|&val| {
                let obj: *mut Object = msg_send![cls, alloc];
                let obj: *mut Object = msg_send![obj, initWithUnsignedInteger:val];
                obj
            })
            .collect();
        
        let array_cls = objc::runtime::Class::get("NSArray").unwrap();
        let array: *mut Object = msg_send![array_cls, alloc];
        let array: *mut Object = msg_send![array,
            initWithObjects:numbers.as_ptr()
            count:numbers.len()
        ];
        
        // Release NSNumber objects
        for num in numbers {
            let _: () = msg_send![num, release];
        }
        
        array
    }
}

/// Depthwise convolution operations for MPSGraph
impl MPSGraph {
    /// Creates a 2D depthwise convolution operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `source` - A 2D image source tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `weights` - Weights tensor (rank=4, layout defined by descriptor.weightsLayout)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn depthwise_convolution_2d(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        descriptor: &MPSGraphDepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution2DWithSourceTensor:source.0
                weightsTensor:weights.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D depthwise convolution gradient for data operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 2D input gradient tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `weights` - Weights tensor (rank=4, layout defined by descriptor.weightsLayout)
    /// * `output_shape` - Shape of the output tensor (and input tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to data
    pub fn depthwise_convolution_2d_data_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphDepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution2DDataGradientWithIncomingGradientTensor:incoming_gradient.0
                weightsTensor:weights.0
                outputShape:output_shape.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 2D depthwise convolution gradient for weights operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 2D input gradient tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `source` - A 2D image source tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `output_shape` - Shape of the output tensor (and weight tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn depthwise_convolution_2d_weights_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphDepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution2DWeightsGradientWithIncomingGradientTensor:incoming_gradient.0
                sourceTensor:source.0
                outputShape:output_shape.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 3D depthwise convolution operation.
    ///
    /// # Arguments
    ///
    /// * `source` - A 3D image source tensor (at least rank=4, CDHW when channelDimensionIndex = -4)
    /// * `weights` - Weights tensor (rank=4, axes interpreted as CDHW when channelDimensionIndex = -4)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the result
    pub fn depthwise_convolution_3d(
        &self,
        source: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        descriptor: &MPSGraphDepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution3DWithSourceTensor:source.0
                weightsTensor:weights.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 3D depthwise convolution gradient for data operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 3D input gradient tensor (at least rank=4, CDHW)
    /// * `weights` - Weights tensor (rank=4, axes interpreted as CDHW)
    /// * `output_shape` - Shape of the output tensor (and input tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to data
    pub fn depthwise_convolution_3d_data_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        weights: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphDepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution3DDataGradientWithIncomingGradientTensor:incoming_gradient.0
                weightsTensor:weights.0
                outputShape:output_shape.0
                descriptor:descriptor.0
                name:name_obj
            ];
            
            let tensor: *mut Object = msg_send![tensor, retain];
            MPSGraphTensor(tensor)
        }
    }
    
    /// Creates a 3D depthwise convolution gradient for weights operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 3D input gradient tensor (at least rank=4, NCDHW)
    /// * `source` - A 3D image source tensor (at least rank=4, NCDHW)
    /// * `output_shape` - Shape of the output tensor (and weight tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new MPSGraphTensor containing the gradient with respect to weights
    pub fn depthwise_convolution_3d_weights_gradient(
        &self,
        incoming_gradient: &MPSGraphTensor,
        source: &MPSGraphTensor,
        output_shape: &MPSShape,
        descriptor: &MPSGraphDepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> MPSGraphTensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).0,
            None => std::ptr::null_mut(),
        };
        
        unsafe {
            let tensor: *mut Object = msg_send![
                self.0,
                depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:incoming_gradient.0
                sourceTensor:source.0
                outputShape:output_shape.0
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
    use std::collections::HashMap;
    
    #[test]
    fn test_depthwise_convolution_2d() {
        if should_skip_test("test_depthwise_convolution_2d") {
            return;
        }
        
        let graph = MPSGraph::new();
        
        // Create a 1x2x4x4 input (N=1, C=2, H=4, W=4)
        let source_shape = MPSShape::from_slice(&[1, 2, 4, 4]);
        let source_data = vec![
            // Channel 1
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            // Channel 2
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ];
        let source = graph.placeholder(&source_shape, MPSDataType::Float32, None);
        
        // Create a 2x1x3x3 kernel (C=2, multiplier=1, H=3, W=3)
        let weights_shape = MPSShape::from_slice(&[2, 1, 3, 3]);
        let weights_data = vec![
            // Filter for channel 1
            1.0f32, 1.0, 1.0,
            1.0, 2.0, 1.0,
            1.0, 1.0, 1.0,
            // Filter for channel 2
            0.5f32, 0.5, 0.5,
            0.5, 1.0, 0.5,
            0.5, 0.5, 0.5
        ];
        let weights = graph.placeholder(&weights_shape, MPSDataType::Float32, None);
        
        // Create descriptor
        let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor::new_with_layouts(
            TensorNamedDataLayout::NCHW,
            TensorNamedDataLayout::NCHW
        );
        descriptor.set_stride_in_x(1);
        descriptor.set_stride_in_y(1);
        descriptor.set_dilation_rate_in_x(1);
        descriptor.set_dilation_rate_in_y(1);
        descriptor.set_explicit_padding(1, 1, 1, 1);
        descriptor.set_padding_style(PaddingStyle::Explicit);
        
        // Create the depthwise convolution operation
        let result = graph.depthwise_convolution_2d(
            &source,
            &weights,
            &descriptor,
            Some("depthwise_conv")
        );
        
        // Run the graph
        let mut feeds = HashMap::new();
        feeds.insert(&source, crate::MPSGraphTensorData::new(&source_data, &[1, 2, 4, 4], MPSDataType::Float32));
        feeds.insert(&weights, crate::MPSGraphTensorData::new(&weights_data, &[2, 1, 3, 3], MPSDataType::Float32));
        
        let results = graph.run(feeds, &[&result]);
        
        // Get the result and verify shape is [1, 2, 4, 4] (same as input with padding=1)
        let result_data = results[&result].to_vec::<f32>();
        assert_eq!(result_data.len(), 32); // 1*2*4*4=32 elements expected
    }
}